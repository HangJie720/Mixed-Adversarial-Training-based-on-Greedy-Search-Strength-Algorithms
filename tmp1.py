import torch
import numpy as np
from torch.autograd import Variable
a=torch.randn(3,3)
b=torch.randn(3,3)
print a,b
a=Variable(a, requires_grad=True)
b=Variable(b, requires_grad=True)
c=torch.mul(a,b)
c.backward(c.data)
print a.grad