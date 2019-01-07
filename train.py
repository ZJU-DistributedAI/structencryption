import torchvision as tv
import torchvision.transforms as transforms
import torch as t
import numpy as np
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
use_gpu = False 
saved_model_name = ''
name = 'test.cpk' #name to save ckp

transform = transforms.Compose([transforms.ToTensor(),#转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                                ])
path = '~/datas/fashion-mnist-master/data'
trainset=tv.datasets.MNIST(root=path,
                             train=True,
                             download=False,
                             transform=transform)
trainloader=t.utils.data.DataLoader(trainset,
                                    batch_size=16,
                                    shuffle=False,
                                    num_workers=0)
testset=tv.datasets.MNIST(root=path,
                             train=False,
                             download=False,
                             transform=transform)
testloader=t.utils.data.DataLoader(testset,
                                   batch_size=100,
                                   shuffle=True,
                                   num_workers=0)


import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class Fake_link_type1(nn.Module):
    def __init__(self,inplanes,outplanes,stride=1,zero_init = True):
        super(Fake_link_type1,self).__init__()
        self.conv1 = conv3x3(inplanes,outplanes,stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes,outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.stride = stride
        if zero_init:
            zeros = t.Tensor(np.zeros([inplanes,outplanes,3,3]))
            self.conv1.weight = t.nn.Parameter(zeros)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x))
        #bn2 bias 
        return out 

class Fake_link_type2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, zero_init=True):
        super(Fake_link_type2, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
        if zero_init :
            zeros = t.Tensor(np.zeros([planes]))
            self.bn2.weight = t.nn.Parameter(zeros)
            self.bn2.bias = t.nn.Parameter(zeros)
            zeros_bn3 = t.Tensor(np.zeros([planes *self.expansion]))
            self.bn3.bias = t.nn.Parameter(zeros_bn3)
            #bn3 bias is remained

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,3)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*4*4,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)   
        #self.fakelink1 = Fake_link_type1(6,6)
        #self.fakelink2 = Fake_link_type2(16,4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        #x = x + self.fakelink1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        #x = x + self.fakelink2(x)
        x = x.view(-1, 16 * 4 * 4)
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
    net.load_state_dict(t.load(saved_model_name))
import time

start_time = time.time()
for epoch in range(10):
    if name:
        t.save(net.state_dict(),name)
    running_loss=0.0
    sum_acc = 0.0
    net.train()
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

    net.eval()
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


