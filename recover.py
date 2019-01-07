import torchvision as tv
import torchvision.transforms as transforms
import torch as t
import numpy as np
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
use_gpu = True 
saved_model_name = 'fake_init'
name = ''

transform = transforms.Compose([transforms.ToTensor(),#转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                                ])
#训练集
trainset=tv.datasets.CIFAR10(root='~/data/',
                             train=True,
                             download=False,
                             transform=transform)

trainloader=t.utils.data.DataLoader(trainset,
                                    batch_size=16,
                                    shuffle=False,
                                    num_workers=0)
#测试集
testset=tv.datasets.CIFAR10(root='~/data/',
                             train=False,
                             download=False,
                             transform=transform)

testloader=t.utils.data.DataLoader(testset,
                                   batch_size=100,
                                   shuffle=True,
                                   num_workers=0)


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

        #self.fake_conv = nn.Conv2d(6,6,3,padding=1)
        #self.fake_bn = nn.BatchNorm2d(6)
        #self.fake_relu = nn.ReLU(inplace = True)

        #zeros = t.Tensor(np.zeros([6,6,3,3]))
        #self.fake_conv.weight = t.nn.Parameter(zeros)
        #zeros_b = t.Tensor(np.zeros([6]))
        #self.fake_conv.bias = t.nn.Parameter(zeros_b)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        #x = x + self.fake_relu(self.fake_bn(self.fake_conv(x)))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)
        return x

from torch.autograd import Variable

net=Net()
if use_gpu:
    net = net.cuda()
print(net)

from torch import optim
criterion=nn.CrossEntropyLoss()
learning_rate = 0.001 
optimizer=optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9)

if saved_model_name :
    parameters = t.load(saved_model_name)
    del parameters['fake_conv.weight']
    del parameters['fake_conv.bias']
    del parameters['fake_bn.running_mean']
    del parameters['fake_bn.running_var']
    del parameters['fake_bn.weight']
    del parameters['fake_bn.bias']
    del parameters['fake_bn.num_batches_tracked']
    net.load_state_dict(parameters)
import time

start_time = time.time()
for epoch in range(10):
    if name:
        t.save(net.state_dict(),name)
    running_loss=0.0
    sum_acc = 0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs,labels=Variable(inputs),Variable(labels)
        optimizer.zero_grad()

        outputs=net(inputs)
        loss=F.nll_loss(outputs,labels)

        ma,ind = t.max(outputs,1)
        acc = t.sum(t.eq(ind,labels)).cpu().numpy()
        sum_acc += 1.0 * acc / outputs.size(0)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
            print('acc:' + str(sum_acc/10)+ '%')
            sum_acc = 0.0

    sum_acc = 0.0
    for i,data in enumerate(testloader,0):
        inputs,labels=data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs,labels=Variable(inputs),Variable(labels)
        outputs=net(inputs)
        optimizer.zero_grad()
        ma,ind = t.max(outputs,1)
        acc = t.sum(t.eq(ind,labels)).cpu().numpy()
        sum_acc += 1.0 * acc / outputs.size(0)
    print('test acc:' + str(sum_acc)+ '%')
print('finished training')
end_time = time.time()
print("Spend time:", end_time - start_time)


