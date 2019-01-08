import numpy as np
from numpy import *
from pylab import *
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt

acc0=np.load('/home/hankeji/Desktop/RFN/GTSRB/GTSRB_Dropout_0.6.npy')
acc1=np.load('/home/hankeji/Desktop/RFN/GTSRB/GTSRB_Dropout_0.7.npy')
acc2=np.load('/home/hankeji/Desktop/RFN/GTSRB/GTSRB_Dropout_0.8.npy')
acc3=np.load('/home/hankeji/Desktop/RFN/GTSRB/GTSRB_Dropout_0.9.npy')
acc4=np.load('/home/hankeji/Desktop/RFN/GTSRB/GTSRB_ORI.npy')


'''
acc0=np.load('/home/hankeji/Desktop/RFN/mnist_0.4/mnist_SWFN_0.2.npy')
acc1=np.load('/home/hankeji/Desktop/RFN/mnist_0.4/mnist_RFN_0.4.npy')
acc2=np.load('/home/hankeji/Desktop/RFN/mnist_0.4/mnist_FSFN_0.4.npy')
acc3=np.load('/home/hankeji/Desktop/RFN/mnist_0.4/mnist_Dropout_0.4.npy')
acc4=np.load('/home/hankeji/Desktop/RFN/mnist_ORI.npy')
'''
'''
acc0=np.load('/home/hankeji/Desktop/mnist_Dropout_0.6.npy')
acc1=np.load('/home/hankeji/Desktop/mnist_Dropout_0.7.npy')
acc2=np.load('/home/hankeji/Desktop/mnist_Dropout_0.8.npy')
acc3=np.load('/home/hankeji/Desktop/mnist_Dropout_0.9.npy')
acc4=np.load('/home/hankeji/Desktop/RFN/mnist_ORI.npy')
'''

plot01,=plt.plot(acc4[:,0],acc4[:,2],'c:',label='ori')
#plot1,=plt.plot(acc1[:,0],acc1[:,1],'y',label='FSFN_ori')
plot2,=plt.plot(acc0[:,0],acc0[:,2],'y.',label='Dropout_0.6')
#plot3,=plt.plot(acc2[:,0],acc2[:,1],'k',label='SWFN_ori')
plot4,=plt.plot(acc1[:,0],acc1[:,2],'k.',label='Dropout_0.7')
#plot5,=plt.plot(acc3[:,0],acc3[:,1],'m',label='DROPOUT_ori')
plot6,=plt.plot(acc2[:,0],acc2[:,2],'m.',label='Dropout_0.8')
plot7,=plt.plot(acc3[:,0],acc3[:,2],'b',label='Dropout_0.9')

plt.title('')
plt.legend(handles=[plot01, plot2, plot4, plot6, plot7])
plt.xlabel('phi(scale of gradient added to sample)')
plt.ylabel('accuracy of model')
ax=plt.axes()
plt.show()


'''
a0=acc0[:,1]
a1=acc1[:,1]
a2=acc2[:,1]
a3=acc3[:,1]
#a4=acc4[:,1]
a0=np.reshape(a0,(1,31))
a1=np.reshape(a1,(1,31))
a2=np.reshape(a2,(1,31))
a3=np.reshape(a3,(1,31))
print a1

a0=np.mean(a0,axis=1)
a1=np.mean(a1,axis=1)
a2=np.mean(a2,axis=1)
a3=np.mean(a3,axis=1)


a0=a0.var()
a1=a1.var()
a2=a2.var()
a3=a3.var()
#print a0,a1,a2,a3
'''