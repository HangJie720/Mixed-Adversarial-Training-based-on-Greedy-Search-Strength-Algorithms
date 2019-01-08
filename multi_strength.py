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

import random
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

cit=nn.NLLLoss()

def singel_strength():
    for i in range(18):
        model = torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_0.pkl').cpu()
        print ('*'*45)
        print ('Training {}_th model'.format(i+1))
        #print (i)
        if i < 9:
            eps = [(i+1) * 0.1]
            print ('eps is {}'.format(eps))
            print ('*'*45)
            print ('\n'*2)

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

            torch.save(model,'/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_' + str(eps[0]) + '.pkl')

        else:

            eps = [(i-8) * 0.1]
            print ('eps is {}'.format(eps))
            print ('*' * 45)
            print ('\n' * 2)
            model = model_list[0].cpu()
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = Variable(data, requires_grad=True), Variable(target)
                optimizer0=optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
                output = model(data)
                loss = cit(output, target)
                optimizer0.zero_grad()
                loss.backward()
                optimizer0.step()


                data, target = Variable(data.data, requires_grad=True), Variable(target.data)
                attack_model_index = random.sample([4], 1)[0]
                attack_model = model_list[attack_model_index].cpu()
                #optimizer1=optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=1e-4)
                adv = symbolic_fgs(attack_model, cit, data, target, eps=eps)

                data = Variable(adv.data, requires_grad=True)
                output1 = model(data)
                optimizer0.zero_grad()
                loss = cit(output1, target)
                loss.backward()
                optimizer0.step()
            torch.save(model, '/home/hankeji/Desktop/model_GTSRB/GTSRB_single_model_' + str(eps[0]) + '.pkl')


def mix_strength():
    #model=model_list[0].cpu()
    #a=b1(1, train_model)
    #print (a)
    eps0 =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #eps0=[0.2, 0.6, 0.9]
    #eps0=search_strength()
    #tmp_n = np.random.randint(1, 9, 1)[0]
    eps0=random.sample(eps0, 3)
    tmp=len(eps0)
    print ('\n'*2)
    print (tmp)
    train_model = torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_0.pkl').cpu()
    for i in range(10):
        eps=[eps0[i%3]]

        for batch_idx, (data, target) in enumerate(train_loader):

            print (batch_idx)
            data, target = Variable(data, requires_grad=True), Variable(target)

            optimizer0 = optim.Adam(train_model.parameters(), lr=0.01, weight_decay=1e-4)
            output = train_model(data)
            loss = cit(output, target)
            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()

            attack_model_index = random.sample([0, 1, 2, 4], 1)[0]
            attack_model = model_list[attack_model_index].cpu()
            #optimizer1 = optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=1e-4)
            adv = symbolic_fgs(attack_model, cit, data, target, eps=eps)
            data = Variable(adv.data, requires_grad=True)
            output1 = train_model(data)
            optimizer0.zero_grad()
            loss = cit(output1, target)
            loss.backward()
            optimizer0.step()

    torch.save(train_model, '/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_random_strength.pkl')


def defense_eval():
    acc=[]
    acc=np.asarray(acc, np.float)
    a=31
    attack_model_index = random.sample([3], 1)[0]
    attack_model = model_list[attack_model_index].cpu()
    for i in range(9):
        tmp_model=torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_' + str((i+1)*0.1) + '.pkl').cpu()
        #acc=np.append(acc, i)
        for i in range(a):
            if i == 0:
                eps = [0]
            else:
                eps = [(i) * 0.03]
            acc=np.append(acc, eps)

            corr=0
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx>0:
                    break
                print (batch_idx)
                data, target = Variable(data, requires_grad=True), Variable(target)
                data=symbolic_fgs(attack_model, cit, data, target, eps=eps)
                data=Variable(data.data, requires_grad=True)
                output=tmp_model(data)
                pred=output.data.max(1)[1]
                corr+=pred.eq(target.data).cpu().sum()
            #print (batch_idx+1)
            corr=corr/((batch_idx)*batch_size)
            acc=np.append(acc, corr)
    acc=np.reshape(acc, (9,a,2))
    #print acc
    np.save('/home/hankeji/Desktop/jsai/data/GTSRB_model_acc_single_modle_clean.npy', acc)#_single model _mix_strength [0.1,0.6,0.8]
    return acc


def search_strength():
    attack_index=random.sample([3], 1)[0]
    attack_model=model_list[attack_index].cpu()
    strength_list=[]
    strength_list=np.asarray(strength_list, np.float)
    for i in range(9):
        print ('Searching scale of {}!'.format((i+1)*0.1))
        eps=[(i+1)*0.1]
        tmp_acc=[]
        tmp_acc=np.asarray(tmp_acc, np.float)
        corr=0
        for j in range(9):
            tmp_model = torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_' + str((j + 1) * 0.1) + '.pkl').cpu()
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target=Variable(data, requires_grad=True), Variable(target)
                if batch_idx>0:
                    break
                adv=symbolic_fgs(attack_model, cit, data, target, eps=eps)
                data=Variable(adv.data, requires_grad=True)

                output=tmp_model(data)
                pred=output.data.max(1)[1]

                corr+=pred.eq(target.data).sum()
            corr=corr/batch_size
            tmp_acc=np.append(tmp_acc, corr)
            tmp_acc=np.reshape(tmp_acc, (-1, 1))
        tmp_acc=torch.from_numpy(tmp_acc)
        tmp_acc_idx=tmp_acc.max(0)[1]
        aa=float((tmp_acc_idx.numpy()+1))*0.1
        print (aa)
        strength_list=np.append(strength_list, float((tmp_acc_idx.numpy()+1))*0.1)
    final_strength_list=[]
    final_strength_list=np.asarray(final_strength_list, np.float)
    for i in strength_list:
        if i not in final_strength_list:
            final_strength_list=np.append(final_strength_list, i)
    print (final_strength_list)
    return final_strength_list



def vs():
    attack_index = random.sample([3], 1)[0]
    attack_model = model_list[attack_index].cpu()
    mm=torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_all_strength.pkl')
    ms=torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_0.3.pkl')
    m_all=torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_all_strength.pkl')
    m_random=torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_multi_model_random_strength.pkl')
    m_sm=torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_single_model_multi_strength.pkl')
    m_ss=torch.load('/home/hankeji/Desktop/model_GTSRB/GTSRB_single_model_0.3.pkl')
    acc=[]
    acc=np.asarray(acc, np.float)
    a=31
    for i in range(a):
        acc_mm = 0
        acc_ms = 0
        acc_m_all = 0
        acc_m_random=0
        acc_sm=0
        acc_ss=0
        if i==0:
            eps=[0]
        else:
            eps=[(i+1)*0.03]

        for batch_idx, (data, target) in enumerate(test_loader):
            #print (data[0])
            if batch_idx>0:
                break
            print ('Here coming {}_th batch'.format(batch_idx + 1))
            data, target = Variable(data, requires_grad=True), Variable(target)
            adv=symbolic_fgs(attack_model, cit, data, target, eps)
            data=Variable(adv.data, requires_grad=True)


            output1 = mm(data)
            pred1 = output1.data.max(1)[1]
            acc_mm += pred1.eq(target.data).sum()

            output2 = ms(data)
            pred2 = output2.data.max(1)[1]
            acc_ms += pred2.eq(target.data).sum()

            output3 = m_all(data)
            pred3 = output3.data.max(1)[1]
            acc_m_all += pred3.eq(target.data).sum()

            output4 = m_random(data)
            pred4 = output4.data.max(1)[1]
            acc_m_random += pred4.eq(target.data).sum()

            output5 = m_sm(data)
            pred5= output5.data.max(1)[1]
            acc_sm += pred5.eq(target.data).sum()

            output6 = m_ss(data)
            pred6 = output6.data.max(1)[1]
            acc_ss += pred6.eq(target.data).sum()

        acc_mm = acc_mm / batch_size#len(test_loader.dataset)
        acc_ms = acc_ms /batch_size# len(test_loader.dataset)
        acc_m_all= acc_m_all/batch_size#len(test_loader.dataset)
        acc_m_random=acc_m_random/batch_size#len(test_loader.dataset)
        acc_sm=acc_sm/batch_size#len(test_loader.dataset)
        acc_ss=acc_ss/batch_size#len(test_loader.dataset)

        acc=np.append(acc, eps)
        acc=np.append(acc, acc_mm)
        acc=np.append(acc, acc_ms)
        acc=np.append(acc, acc_m_all)
        acc=np.append(acc, acc_m_random)
        acc = np.append(acc, acc_sm)
        acc = np.append(acc, acc_ss)

    acc=np.reshape(acc, (a, -1))
    np.save('/home/hankeji/Desktop/jsai/data/GTSRB_mmVSms_7.npy', acc)
    return acc


def vs_car():#variant strength carlini
    mm = torch.load('/home/hankeji/Desktop/jsai/models/multi_modle_multi_strength_VS_10.pkl')
    ms = torch.load('/home/hankeji/Desktop/jsai/models/multi_modle_single_strength_VS.pkl')
    m_all = torch.load('/home/hankeji/Desktop/jsai/models/multi_modle_all_strength_VS.pkl')
    m_random = torch.load('/home/hankeji/Desktop/jsai/models/multi_modle_random_strength_VS.pkl')
    acc_mm = 0
    acc_ms = 0
    acc_m_all = 0
    acc_m_random = 0

    data_car = np.load('/home/hankeji/Desktop/Adversarial Examples/Train-R-FGSM-0.1.npy')
    target_car = np.load('/home/hankeji/Desktop/Adversarial Examples/Label-R-FGSM-0.1.npy')
    data_car = np.asarray(data_car, np.float32)
    data_car = np.reshape(data_car, (-1, 1, 28, 28))
    data_car = Variable(torch.from_numpy(data_car), requires_grad=True)

    target_car = np.asarray(target_car, np.int64)
    target_car = Variable(torch.from_numpy(target_car))
    target_car = target_car.max(1)[1]
    data=data_car
    target=target_car
    print (data.size())
    print (target.size())

    output1 = mm(data)
    pred1 = output1.data.max(1)[1]
    acc_mm += pred1.eq(target.data).sum()

    output2 = ms(data)
    pred2 = output2.data.max(1)[1]
    acc_ms += pred2.eq(target.data).sum()

    output3 = m_all(data)
    pred3= output3.data.max(1)[1]
    acc_m_all += pred3.eq(target.data).sum()

    output4 = m_random(data)
    pred4 = output4.data.max(1)[1]
    acc_m_random+= pred4.eq(target.data).sum()

    print ('Accuracy of mm is {}'.format(acc_mm/500))
    print ('Accuracy of ms is {}'.format(acc_ms / 500))
    print ('Accuracy of m_all is {}'.format(acc_m_all / 500))
    print ('Accuracy of m_random is {}'.format(acc_m_random / 500))

def ori_attack():
    model=torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_submodels_0.pkl').cpu()
    corr=0
    #corr_du=0
    #corr_ori=0
    acc=[]
    acc=np.asarray(acc, np.float)
    attack_index = random.sample([3], 1)[0]
    attack_model = model_list[attack_index].cpu()
    a=31
    for i in range(a):
        if i==0:
            eps=[0]
        else:
            eps=[(i)*0.03]
        print ('*'*45)
        print('eps is {}'.format(eps[0]))
        print ('*'*45)
        print ('\n'*2)
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx > 0:
                break
            data, target = Variable(data, requires_grad=True), Variable(target)
            adv=symbolic_fgs(attack_model, cit, data, target, eps=eps)
            data=Variable(adv.data, requires_grad=True)

            output=model(data)
            pred=output.data.max(1)[1]
            corr+=pred.eq(target.data).sum()


        corr = corr/(batch_idx*batch_size)
        acc=np.append(acc, eps)
        acc=np.append(acc, corr)

    acc=np.reshape(acc, (-1, 2))
    np.save('/home/hankeji/Desktop/jsai/data/GTSRB_ori.npy', acc)
    return acc


def b1(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:


        data, target = Variable(data, requires_grad=True), Variable(target)
        output = model(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    return 100. * correct / len(test_loader.dataset)

def bgr2rgb(a):
    tmp=np.ones((32,32,3), np.float32)
    tmp[: , :, 0]=a[2]
    tmp[:, :, 1]=a[1]
    tmp[:, :, 2]=a[0]
    return tmp

if __name__=='__main__':
    #singel_strength()

    #mix_strength()

    #acc1=np.load('/home/hankeji/Desktop/jsai/data/mnist_ori.npy')
    '''
    import matplotlib.pyplot as plt
    acc=defense_eval()
    acc1=ori_attack()
    #acc=np.load('/home/hankeji/Desktop/jsai/data/GTSRB_model_acc_single_modle_30.npy')
    #acc1=np.load('/home/hankeji/Desktop/jsai/data/mnist_ori.npy')
    #acc=defense_eval()
    plot1, = plt.plot(acc[0,:, 0], acc[0,:, 1], linewidth=2.5, color='b',label='0.1')
    plot2, = plt.plot(acc[1,:, 0], acc[1,:, 1], linewidth=2.5, color='b', marker='p', label='0.2')
    plot3, = plt.plot(acc[2,:, 0], acc[2,:, 1], linewidth=2.5, color='y',  label='0.3')
    plot4, = plt.plot(acc[3,:, 0], acc[3,:, 1], linewidth=2.5, color='y',marker='p',label='0.4')
    plot5, = plt.plot(acc[4,:, 0], acc[4,:, 1], linewidth=2.5, color='k',  label='0.5')
    plot6, = plt.plot(acc[5,:, 0], acc[5,:, 1], linewidth=2.5, color='k',marker='p',label='0.6')
    plot7, = plt.plot(acc[6,:, 0], acc[6,:, 1], linewidth=2.5, color='r',label='0.7')
    plot8, = plt.plot(acc[7,:, 0], acc[7,:, 1], linewidth=2.5, color='r', marker='p', label='0.8')
    plot9, = plt.plot(acc[8,:, 0], acc[8,:, 1], linewidth=2.5, color='g',label='0.9')
    plot10, =  plt.plot(acc1[:, 0], acc1[:, 1], linewidth=2.5, color='g', marker='p', label='ori')

    #plt.title('Single strength ',fontsize=15, fontstyle='oblique')
    plt.legend(handles=[plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10])
    plt.xlabel('$\phi$ (scale of gradient added to sample)', fontsize=15, fontstyle='oblique')
    plt.ylabel('Accuracy of model')
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    plt.show()
    '''


    '''
    import matplotlib.pyplot as plt
    #acc=np.load('/home/hankeji/Desktop/jsai/data/GTSRB_mmVSms.npy')
    acc=vs()
    #acc1=ori_attack()
    acc1=np.load('/home/hankeji/Desktop/jsai/data/GTSRB_ori.npy')
    plot1, = plt.plot(acc[:, 0], acc[:, 1], linewidth=2.5, color='r', marker='>',label='mm')
    plot2, = plt.plot(acc[:, 0], acc[:, 2], linewidth=2.5, color='g', label='ms')
    plot3, = plt.plot(acc[:, 0], acc[:, 3], linewidth=2.5, color='b', label='m_all')
    plot4, = plt.plot(acc[:, 0], acc[:, 4], linewidth=2.5, color='y', marker='>',label='m_ramdom')
    plot5, = plt.plot(acc[:, 0], acc[:, 5], linewidth=2.5, color='k', label='sm')
    plot6, = plt.plot(acc[:, 0], acc[:, 6], linewidth=2.5, color='k', marker='p', label='ss')
    plot7, = plt.plot(acc1[:, 0], acc1[:, 1], linewidth=2.5, color='b', marker='p', label='ori')
    # plt.title('mm VS ms', fontsize=15, fontstyle='oblique')
    plt.legend(handles=[plot1, plot2, plot3, plot4, plot5, plot6, plot7])
    plt.xlabel('$\phi$ (scale of gradient added to sample)', fontsize=15, fontstyle='oblique')
    plt.ylabel('Accuracy of model')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.show()
    '''






#*************************
# ce du, God help!       *
#*************************

    '''
    import matplotlib.pyplot as plt
    # acc=np.load('/home/hankeji/Desktop/jsai/data/GTSRB_mmVSms.npy')
    #acc = vs()
    acc = ori_attack()
    plot1, = plt.plot(acc[:, 0], acc[:, 1], linewidth=2.5, color='r', label='ori')
    plot2, = plt.plot(acc[:, 0], acc[:, 2], linewidth=2.5, color='b', marker='>', label='du')
    plot3, = plt.plot(acc[:, 0], acc[:, 3], linewidth=2.5, color='k', label='0.9')
    plt.legend(handles=[plot1, plot2, plot3])
    plt.xlabel('$\phi$ (scale of gradient added to sample)', fontsize=15, fontstyle='oblique')
    plt.ylabel('Accuracy of model')
    #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    #plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.show()
    '''


#*****************************
#                            *
#   Evaluate test accuracy!  *
#                            *
#*****************************
    '''
    mm = torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_multi_modle_multi_strength_VS.pkl')# test_accuracy
    ms = torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_multi_modle_single_strength_VS.pkl')
    m_all = torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_multi_modle_all_strength_VS.pkl')
    m_random = torch.load('/home/hankeji/Desktop/jsai/models/GTSRB_multi_modle_random_strength_VS.pkl')
    eval_list=[mm, ms, m_all, m_random]
    #mix_strength()
    for i in range(4):
        model=eval_list[i]
        a=b1(1)
        print (a)
    '''
