from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random

#eps_list=[0.1, 0.6, 0.8]
def symbolic_fgs(model,cit, data, target, eps, clipping=True):
    """
    FGSM attack.
    """
    eps0=random.sample(eps, 1)[0]

    #optimizer1 = optim.Adam(model.parameters(), lr=0.01)
    logits = model(data)

    loss = cit(logits, target)
    loss.backward()
    grad = data.grad
    grad=torch.sign(grad)
    # Add perturbation to original example to obtain adversarial example
    adv_x =(data + eps0*grad)

    if clipping:
        adv_x = torch.clamp(adv_x, 0, 1)
    return adv_x


def iter_fgs(model, x, y, steps, eps):
    """
    I-FGSM attack.
    """
    model=model.cpu()
    adv_x = x
    # iteratively apply the FGSM with small step size
    #optimizer=optim.Adam(model.parameters(), lr=0.01)
    for i in range(steps):

        logits = model(adv_x)
        #optimizer.zero_grad()
        loss=F.nll_loss(logits, y)
        loss.backward()
        grad=x.grad
        adv_x = symbolic_fgs(adv_x, grad, eps, True)
        adv_x=Variable(adv_x.data, requires_grad=True)
    return adv_x

def least_likly(model, cit, data, target, eps=0.3):
    #eps = random.sample(eps_list, 1)[0]
    tmp=model(data)
    tmp=-tmp
    loss=cit(tmp, target)
    loss.backward()
    grad = data.grad
    adv_x = data+torch.mul(torch.sign(grad),eps)
    adv_x = torch.clamp(adv_x, 0 ,1)
    return adv_x

def It_least_likly(model, data, target, steps, eps):
    for i in range(steps):
        tmp = model(data)
        tmplabel = tmp.data.min(1)[1]
        loss = F.nll_loss(tmplabel, target)
        loss.backward()
        grad = data.grad
        data=data+torch.mul(torch.sign(grad), eps/steps)
    return torch.sign(grad)
