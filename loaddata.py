import numpy as np
import torch
import torchvision.transforms as transforms
from PIL.Image import *
import cv2
data=np.load('/home/hankeji/Desktop/papercode/GTSRB/traindata/data_scale.npy')
labels=np.load('/home/hankeji/Desktop/papercode/GTSRB/traindata/labels.npy')

print data.shape
print labels.shape
for i in range(data.shape[0]):
   data[i]=data[i]*255
    #Image.show(tmp)
    #print tmp

data=np.asarray(data, np.uint8)
labels=np.asarray(labels,np.int64)
print labels[0]

for i in range(labels.shape[0]):
    print type(labels[i])
    print labels[i]
    labels[i]=((labels[i]))
    print labels[i]
    print type(labels[i])
np.save('/home/hankeji/Desktop/papercode/GTSRB/traindata/labels.npy',labels)
np.save('/home/hankeji/Desktop/papercode/GTSRB/traindata/data_scale.npy',data)
