from __future__ import division
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import cv2
import  copy
import random
import gc
from attack_means import symbolic_fgs, least_likly
batch_size=1024
#import cleverhans as clh
# Training settings
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
        datasets.MYDATA('/home/hankeji/Desktop/cifar12', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size= batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
        datasets.MYDATA('/home/hankeji/Desktop/cifar12', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size= batch_size, shuffle=True, **kwargs)


from search_networks import model0, model1,model2, model3 ,model4
model0_=model0()
model1_=model1()
model2_=model2()
model3_=model3()
model4_=model4()
model0_=torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_0.pkl')
model1_=torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_1.pkl')
model2_=torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_2.pkl')
model3_=torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_3.pkl')
model4_=torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_4.pkl')
model_list=[model0_, model1_, model2_, model3_, model4_]
train_model=model_list[0]

cit=nn.NLLLoss()
def singel_strength():
    for i in range(9):
        print ('*'*45)
        print ('Training {}_th model'.format(i+1))
        #print (i)

        if i < 9:
            eps = [0.9]
            print ('eps is {}'.format(eps))
            print ('*'*45)
            print ('\n'*2)
            model=torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_0.pkl').cpu()

            print (id(model))
            print (id(train_model))
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data, requires_grad=True), Variable(target)
                optimizer0 = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
                output =model(data)
                loss = cit(output, target)
                optimizer0.zero_grad()
                loss.backward()
                optimizer0.step()

                attack_model_index = random.sample([0, 1, 2, 4], 1)[0]
                attack_model = model_list[attack_model_index].cpu()
                optimizer1 = optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=1e-4)
                adv=symbolic_fgs(attack_model, cit, data, target, eps=eps)
                data = Variable(adv.data, requires_grad=True)
                output1 = model(data)
                optimizer0.zero_grad()
                loss = cit(output1, target)
                loss.backward()
                optimizer0.step()
            torch.save(model,'/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_du_test.pkl')
            del(model)
            gc.collect()

    #torch.save(model, '/home/hankeji/Desktop/jsai/models/GTSRB_multi_model_du.pkl')
if __name__=='__main__':
    singel_strength()