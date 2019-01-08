import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from torch.autograd import Variable

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/hankeji/Desktop/papercode/tmp/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=128, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/hankeji/Desktop/papercode/tmp/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=128, shuffle=True, **kwargs)

from emu import Net,optimizer
model=Net()
model.load_state_dict(torch.load('/home/hankeji/Desktop/mnist_test.pkl'))
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv2_drop = model.conv2_drop
        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.norm1=model.norm1
        self.norm2=model.norm2
        self.norm3=model.norm3
        self.norm4=model.norm4
        self.norm5=model.norm5
    def forward(self, x):
        x=self.conv1(self.norm1(x))
        print (x.size())
        x = F.relu(F.max_pool2d(x, 2))
        print(x.size())
        x=self.conv2(self.norm2(x))
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))
        print(x.size())
        x = x.view(-1, 320)
        x = F.relu(self.fc1(self.norm4(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(self.norm5(x))
        return F.log_softmax(x)
model1=Net1()
model1.cuda()
model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)

    output = model1(data)
    test_loss += F.nll_loss(output, target).data[0]
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
