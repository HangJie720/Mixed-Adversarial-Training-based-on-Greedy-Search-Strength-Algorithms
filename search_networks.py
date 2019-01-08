import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
        datasets.MYDATA('/home/hankeji/Desktop/cifar12', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=128, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(
        datasets.MYDATA('/home/hankeji/Desktop/cifar12', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=128, shuffle=True, **kwargs)


class model0(nn.Module):
    def __init__(self):
        super(model0, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 172)
        self.fc2 = nn.Linear(172, 43)
        self.norm1=nn.BatchNorm2d(3)
        self.norm2=nn.BatchNorm2d(10)
        self.norm3=nn.BatchNorm2d(20)
        self.norm4=nn.BatchNorm1d(500)
        self.norm5=nn.BatchNorm1d(172)
    def feature(self, x):
        x = F.relu(F.max_pool2d(self.conv1(self.norm1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(self.norm2(x))), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(self.norm4(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(self.norm5(x))
        return x
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(self.norm1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(self.norm2(x))), 2))
        #print x.size()
        x = x.view(-1, 500)
        x = F.relu(self.fc1(self.norm4(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(self.norm5(x))
        x=F.log_softmax(x)
        #print (x.size())
        return x
class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def feature(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d((self.conv2(x)), 2)

        x = F.avg_pool2d(x, 2)
        # print x.size()
        x = x.view(-1, 1, 80)
        x = F.avg_pool1d(x, kernel_size=38, stride=1)
        # print x.size()
        x = x.view(-1, 43)
        return x
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d((self.conv2(x)), 2)

        x = F.avg_pool2d(x, 2)
        #print x.size()
        x = x.view(-1, 1, 80)
        x = F.avg_pool1d(x, kernel_size=38, stride=1)
        #print x.size()
        x = x.view(-1, 43)
        x=F.log_softmax(x)
        #print (x.size())
        return x
class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.con1 = nn.Conv2d(3, 6, kernel_size=5)
        self.con2 = nn.Conv2d(6, 6,5)
        self.con3 = nn.Conv2d(6, 16, 3)
        self.fc=nn.Linear(144,43)

    def feature(self, x):
        x = F.avg_pool2d(self.con1(x), 2)
        x = F.avg_pool2d(self.con2(x), 2)
        x = self.con3(x)
        print x.size()
        x = x.view(-1, 144)
        x = self.fc(x)
        return x
    def forward(self, x):
        x = F.avg_pool2d(self.con1(x), 2)
        x = F.avg_pool2d(self.con2(x), 2)
        x = self.con3(x)
        #print x.size()
        x = x.view(-1, 144)
        x = self.fc(x)
        return F.log_softmax(x)


class model3(nn.Module):
    def __init__(self):
        super(model3, self).__init__()
        self.fc1=nn.Linear(3072,172)
        self.drop=nn.Dropout()
        self.fc2=nn.Linear(86, 43)
        self.pool=nn.AvgPool1d(1, stride=2)
        self.norm1=nn.BatchNorm1d(3072)
        self.norm2=nn.BatchNorm1d(172)
        self.norm3=nn.BatchNorm1d(86)

    def feature(self, x):
        x = x.view(-1, 3072)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.norm2(x)

        x = x.view(-1, 1, 50)
        x = self.pool(x)
        x = x.view(-1, 16)
        x = self.norm3(x)
        x = self.fc2(x)
        return x
    def forward(self ,x):
        x = x.view(-1, 3072)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.norm2(x)
        #print x.size()
        x = x.view(-1, 1, 172)
        x = self.pool(x)
        #print x.size()
        x = x.view(-1, 86)
        x = self.norm3(x)
        x = self.fc2(x)

        x=F.log_softmax(x)
        return x
class model4(nn.Module):
    def __init__(self):
        super(model4, self).__init__()
        self.con1 = nn.Conv2d(3, 64, kernel_size=5)
        self.con2 = nn.Conv2d(64, 64,5)
        self.con3 = nn.Conv2d(64, 16, 3)
        self.fc=nn.Linear(64,10)

    def feature(self, x):
        x = F.avg_pool2d(self.con1(x), 2)
        x = F.avg_pool2d(self.con2(x), 2)
        x = self.con3(x)

        x = x.view(-1, 1, 144)
        x = F.avg_pool1d(x, kernel_size=102, stride=1)
        x = x.view(-1, 43)
        return x
    def forward(self, x):
        x = F.avg_pool2d(self.con1(x), 2)
        x = F.avg_pool2d(self.con2(x), 2)
        x = self.con3(x)

        x = x.view(-1, 1, 144)
        x = F.avg_pool1d(x, kernel_size=102, stride=1)
        x = x.view(-1, 43)
        return F.log_softmax(x)
model_list=[model0(), model1(), model2(), model3(), model4()]


#a-train b-test
def a1(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.6f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def b1(epoch):
    model.eval()
    model.cuda()
    test_loss = 0
    correct = 0
    for data, target in val_loader:

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(val_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return 100. * correct / len(val_loader.dataset)
'''
import numpy as np
import matplotlib.pyplot as plt
c=[]
c=np.asarray(c,np.uint8)
a=15
for i in range(5):
    print ('Here coming modle {}_th'.format(i))
    model=model_list[i]
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    model.cuda()
    for epoch in range(a):
        a1(epoch)
        e = b1(epoch)
        c = np.append(c, epoch)
        c = np.append(c, e)
    torch.save(model, '/home/hankeji/Desktop/jsai/models/GTSRB_submodels_' + str(i) + '.pkl')
'''
