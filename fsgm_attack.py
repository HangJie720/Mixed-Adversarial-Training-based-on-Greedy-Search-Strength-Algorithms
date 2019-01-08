# when call main annoate train phase
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import random
import copy

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
        datasets.MYDATA('/home/hankeji/Desktop/cifar12', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MYDATA('../tmp', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1, shuffle=False, **kwargs)

import argparse
import os
import shutil
import time
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import cv2

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
from search_networks import Net,optimizer

model = Net()
model = torch.load('/home/hankeji/Desktop/papercode/tmp/DeseNet_netbest_GTSRB_40.pkl')
#criterion = nn.CrossEntropyLoss().cuda()
mc=['SWFN','Dropout']
model.cuda()
alpha = 0
acc = []
acc = np.asarray(acc, np.float16)
#for j in range(2):
for i in range(31):  # alpha in range(0,0.3)
    cadv = 0
    cori = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        correct = 0
        correct0 = 0
        print alpha
        # print data.size()[0]
        data = torch.FloatTensor(data)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)
        #print data.size()
        data1 = Variable(torch.FloatTensor(torch.ones((3, 32, 32))), requires_grad=True).cuda()
        #print np.max(data.cpu().data.numpy())
        '''
        c = round(np.random.normal(np.floor(3072 * 0.9), 1)).as_integer_ratio()  # RFN
        a0 = np.arange(0, 3071)
        print c[0]
        a = random.sample(a0, c[0])
        x = np.ones((3072, 1), np.float32)
        for i in range(len(a)):
            b = a[i]
            x[b] = 0
        x = np.reshape(x, (1, 3, 32, 32))
        x = Variable(torch.from_numpy(x), requires_grad=True).cuda()
        '''
        '''
        zero_rate=0.4#FSFN
        if zero_rate > 0.5:
            x = np.zeros((3072, 1), np.float32)
            l = (len(str(zero_rate)) - 2)
            print  l
            lw = np.power(10, l)
            fn = round(lw * (1 - zero_rate))
            if fn == 0:
                space = lw
            else:
                space = np.floor(1 / (1 - zero_rate)).astype(np.int)
            for channel in range(3):
                for i in range(int(np.floor(1024 / lw))):
                    a = np.random.random_integers(i * lw + channel * 1024, (i + 1) * lw - 1 + channel * 1024, 1)
                    forward = a
                    backward = a
                    tmpcount = 0
                    while forward > i * lw:
                        if tmpcount == fn:
                            break
                        else:
                            x[forward] = 1
                            tmpcount += 1
                        forward = forward - space
                    while backward <= (i + 1) * lw:
                        if tmpcount == fn:
                            break
                        else:
                            x[backward] = 1
                            tmpcount += 1
                        backward = backward + space
                j = (i + 1) * lw + channel * 1024
                while j < (channel + 1) * 1024:
                    x[j] = 1
                    j = j + space
        else:
            x = np.ones((3072, 1), np.float32)
            l = (len(str(zero_rate)) - 2)
            lw = np.power(10, l)
            fn = round(lw * (zero_rate))
            space = np.floor(1 / zero_rate).astype(np.int)
            for channel in range(3):
                for i in range(int(np.floor(1024 / lw))):
                    a = np.random.random_integers(i * lw + channel * 1024, (i + 1) * lw - 1 + channel * 1024, 1)
                    forward = a
                    backward = a
                    tmpcount = 0
                    while forward > i * lw:
                        x[forward] = 0
                        tmpcount += 1
                        if tmpcount == fn:
                            break
                        forward = forward - space
                    while backward <= (i + 1) * lw:
                        x[backward] = 0
                        tmpcount += 1
                        if tmpcount == fn:
                            break
                        backward = backward + space
                j = (i + 1) * lw + channel * 1024
                while j < (channel + 1) * 1024:
                    x[j] = 0
                    j = j + space
        x = np.reshape(x, (1,3,32, 32))
        x = Variable(torch.from_numpy(x), requires_grad=True).cuda()
        '''
        '''
        zero_rate = 0.4
        # SWFN
        if zero_rate > 0.5:
            x = np.zeros((3072, 1), np.float32)
            l = (len(str(zero_rate)) - 2)
            lw = np.power(10, l)
            space = np.floor(1 / (1 - zero_rate)).astype(np.int)
            for i in range(3):
                for j in range(int(np.floor(1024 / lw))):
                    nf = int((1 - zero_rate) * lw)
                    a0 = np.arange(j * lw+i*1024, (j + 1) * lw+i*1024 - 1)
                    a = random.sample(a0, nf)
                    for k in range(nf):
                        x[a[k]] = 1
                k = (j + 1) * lw+i*1024
                while k < (i+1)*1024-1:
                    x[k] = 1
                    k = k + space
        else:
            x = np.ones((3072, 1), np.float32)
            l = (len(str(zero_rate)) - 2)
            lw = np.power(10, l)
            space = np.floor(1 / (zero_rate)).astype(np.int)
            for i in range(3):
                for j in range(int(np.floor(1024 / lw))):
                    nf = int((zero_rate) * lw)
                    a0 = np.arange(j * lw+i*1024, (j + 1) * lw+i*1024-1)
                    a = random.sample(a0, nf)
                    for k in range(nf):
                        x[a[k]] = 0
                k = (j + 1) * lw+i*1024
                while k < (i+1)*1024-1:
                    x[k] = 0
                    k = k + space
        x = np.reshape(x, (1, 3, 32, 32))
        x = Variable(torch.from_numpy(x), requires_grad=True).cuda()
        '''
        '''
        x=np.ones((3072,1),np.float32)#dropout feature nullification
        for i in range(x.shape[0]):
            x[i]=np.random.binomial(1,0.6,1)
        x=np.reshape(x,(1,3,32,32))
        x = Variable(torch.from_numpy(x), requires_grad=True).cuda()
        '''
        data1 = torch.mul(data, x)
        output = model(data1)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()#don't updata model
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        cori += pred.eq(target.data).cpu().sum()
        print correct / data.size()[0]
        # print (data.grad)
        data0 = data + torch.mul(torch.sign(data.grad), alpha)
        '''

        if batch_idx==0:
            advs = Variable(torch.randn(1, 3, 32, 32))
            advs1 = Variable(torch.randn(1, 3, 32, 32))
            advs=data
            advs1=data0
        elif batch_idx<8:
            advs=torch.cat((advs,data),0)
            advs1=torch.cat((advs1,data0),0)
        elif batch_idx==8:
            np.save('/home/hankeji/Desktop/RFN/GTSRB/samples/ori/'+'FSFN_'+str(alpha)+'_1.npy',advs.cpu().data.numpy())
            np.save('/home/hankeji/Desktop/RFN/GTSRB/samples/adv/' + 'FSFN_'+str(alpha)+'_1.npy',advs1.cpu().data.numpy())
            print ('sucess!!!')
        else:
            break
        '''
        '''

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale(100),
        ]) 
        from PIL.Image import Image
        if batch_idx%1==0:
            a=data0
            a1=data
            a1=a1.view(3,32,32)
            a=a.view(3,32,32)
            a0=a.cpu().data
            a1=a1.cpu().data
            a0=tf(a0)
            Image.show(a0)
            #print a0.shape
            #a0=np.reshape(a0,(3,32,32))
            #a1=np.reshape(a1,(3,32,3))
            #cv2.imshow('kk',a0)
            #cv2.imshow('kk1', a1)
            #cv2.waitKey(100)
        '''
        # data0=Variable(data0.data,requires_grad=True)

        output0 = model(data0)
        pred0 = output0.data.max(1)[1]
        correct0 += pred0.eq(target.data).cpu().sum()
        cadv += pred0.eq(target.data).cpu().sum()
        print correct0 / data.size()[0]
        print ("\n" * 2)
    # torch.save(model, '/home/hankeji/Desktop/RFN/adv_train.pkl'2)
    ori = cori / len(test_loader.dataset)
    adv = cadv / len(test_loader.dataset)
    acc = np.append(acc, alpha)
    acc = np.append(acc, ori)
    acc = np.append(acc, adv)
    alpha = alpha + 0.01
    print ("ori accracy is {:.4f} ").format(ori)
    print ("adv accracy is {:.4f} ").format(adv)
acc = np.reshape(acc, (-1, 3))
np.save('/home/hankeji/Desktop/RFN/GTSRB/GTSRB_FSFN_0.4.npy', acc)