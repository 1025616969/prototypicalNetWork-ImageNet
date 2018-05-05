import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pylab
from torch.autograd import Variable

data=Variable(torch.FloatTensor([[0.1,0.2,0.4,0.3,0.1],
            [0.05,0.09,0.8,0.1,0.02],
            [0.01,0.9,0.04,0.08,0.07]]))
print(data)
x=np.arange(5)
x=np.tile(x,(3,1))
#print(x)

group_labels=['type 1','type 2','type 3','type 4','type 5']
ymajorLocator= pylab.MultipleLocator(0.1) #将y的刻度设置为0.1的倍数
ymajorFormatter= pylab.FormatStrFormatter('%1.1f') #设置y轴标签文本的格式
yminorLocator = pylab.MultipleLocator(0.01) #将此y轴次刻度标签设置为0.1的倍数
ax=plt.subplot(111)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
ax.yaxis.set_minor_locator(yminorLocator)

for i in range(x.shape[0]):
    ax.plot(x[i],data.data.numpy()[i],label='sample '+str(i))
ax.set_xticks(x[0])
ax.set_xticklabels(group_labels,rotation=0)
ax.legend()
ax.grid()
plt.show()

ax=plt.subplot(111)
loss=[3.5,3.4,3.2,3.05,3.12,3.25,3.35,3.01,3.12]
x=np.arange(9)

ax.plot(x,loss,label="loss graph")
group_labels=[]
for i in range(9):
    group_labels.append("epoch "+str(i+1))
ax.set_xticks(x)
ax.set_xticklabels(group_labels,rotation=0)
ax.legend()
ax.grid()
plt.show()