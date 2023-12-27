'''
前向传播计算数据经过网络的输出
后向传播计算梯度值
'''

from torch import nn
from torch.nn import Module

class Net(Module):
    def __init__(self) -> None:
        super(Net, self).__init__() #调用父类的构造函数

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1)
        self.pool1 = nn.MaxPool2d((2,2))
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        return x

