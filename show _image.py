from __future__ import division
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL.Image import *
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy
def bgr_to_rgb(img_bgr):
    # r,g,b=cv2.split(img_bgr)
    img_rgb = np.zeros(img_bgr.shape, img_bgr.dtype)
    img_rgb[:, :, 0] = img_bgr[:, :, 2]
    img_rgb[:, :, 1] = img_bgr[:, :, 1]
    img_rgb[:, :, 2] = img_bgr[:, :, 0]
    return img_rgb

a=np.load('/home/hankeji/Desktop/papercode/GTSRB/traindata/data_scale.npy')
b=np.load('/home/hankeji/Desktop/papercode/GTSRB/testdata/labels.npy')
print a.shape

data=torch.randn(1,3,32,32)
for i in range(a.shape[0]):
    '''
    print ('*'*30)
    print a[i].shape
    print b[i]
    print data.size()
    print('*'*30)
    print ('\n'*2)
    '''
    #tmp=a[i].transpose((1, 2, 0))
    #print tmp.shape
    #cv2.imshow('kk',tmp)
    #cv2.waitKey(100)
    a[i]=a[i]*255
    print np.max(a[i])

'''
    tmp0=np.asarray(a[i],np.float32)
    tf=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale((32)),
    ])
    tf1=transforms.Compose([
        transforms.ToTensor()
    ])
    if i==0:
        tmp=tf(a[i])
        #print type(tmp)
        tmp=tmp.resize((32,32))
        tmp=tf1(tmp)
        #print type(tmp)
        data=tmp
    else:
        tmp=tf(a[i])
        #print type(tmp)
        tmp = tmp.resize((32, 32))
        tmp = tf1(tmp)
        #print type(tmp)
        data=torch.cat((data,tmp),0)
c=int(data.size()[0]/3)
d=torch.randn(c,3,32,32)
for i in range(c):
    tmp2=data[i*3:(i+1)*3]
    print tmp2.size()
    tmp2=tmp2.view(1,3,32,32)
    d[i]=tmp2
'''
np.save('/home/hankeji/Desktop/papercode/GTSRB/traindata/data_scale.npy',a)
