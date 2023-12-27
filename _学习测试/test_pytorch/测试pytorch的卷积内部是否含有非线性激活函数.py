""" 2022.10.9
疑问：根据鲁鹏老师的所述，神经网络里的卷积是包含非线性激活、偏置的，与普通卷积不一样，但非线性激活包含在卷积内部还是外部指定？
解答：编写代码验证，分别用包含和不包含非线性激活层的网络对同一数据进行运算。
结论：pytorch的卷积层内部不包含非线性激活，原因：↓
含有ReLU层的网络输出为0，
没有ReLU层的网络输出为负数，
且证明了神经网络里面的卷积与numpy、手写计算结果一样。
VGG16的网络结构显示：每个卷积层后都跟了一个relu层，而网页上展示的VGG结构基本都忽略了relu层
resnet一部分卷积后跟batchnorm+relu，另一部分是与原数据相加后再接relu,因此都有relu，（参考：https://www.modb.pro/db/488020）
因为卷积也是线性运算，如果没有非线性激活则会退化。
神经网络的卷积运算为：
卷积+偏置
"""

import torch.nn
import torchvision
from torch import nn
from torch.nn import Module
import cv2
import numpy as np
from torchvision.models import resnet18, vgg16


# 包含Relu的网络
class MyNet_Relu(Module):
    def __init__(self):
        super(MyNet_Relu, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),  # 卷积层
            nn.ReLU()  # 非线性激活，添加ReLU层后，输出的负数变成了0
        )
        pass

    def forward(self, x):
        x = self.Conv(x)
        return x


# 不包含Relu的网络
class MyNet(Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),  # 卷积层
        )
        pass

    def forward(self, x):
        x = self.Conv(x)
        return x


# 数据
d1 = -1 * torch.rand((1, 3, 3))  # type:torch.Tensor
net_relu = MyNet_Relu()
net = MyNet()
# 初始化权重
for m in net_relu.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data = torch.ones((1, 1, 3, 3)) / 9.0
        m.bias.data = torch.tensor([1])  # torch.zeros(1)
        print(m.weight)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data = torch.ones((1, 1, 3, 3)) / 9.0
        m.bias.data = torch.zeros(1)
        print(m.weight)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
out_relu = net_relu(d1)
out = net(d1)
print(f"out_relu={out_relu},out={out}")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# numpy计算卷积
src = d1.numpy()  # type:np.ndarray
src = src.reshape((-1))
kernel = torch.ones((3, 3), dtype=float) / 9.0  # type:np.ndarray
kernel = kernel.reshape((-1))
dst = np.convolve(src, kernel, mode='valid')
pass

# 手写卷积计算
src = src[::-1]
src = src.reshape((1, 9))
kernel = kernel.reshape((9, 1))
s = src.dot(kernel)
pass

net = vgg16()
print(net)
net = torchvision.models.resnet.resnet18()
print(net)
pass