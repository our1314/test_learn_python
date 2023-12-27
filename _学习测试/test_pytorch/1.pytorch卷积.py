import torch.nn.functional
from torch import nn
import torch.functional as F

img = torch.rand((1, 3, 28, 28))  # 数据,四维张量

# 1、卷积类进行卷积（输入通道指：数据的通道数，如数据为3通道则卷积核通道数需为3，输出通道指前面卷积核的个数）
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=2)
# 上面语句表示，用64个3x3x3的卷积核对3通道图像进行卷积，得到WxHx64的特征相应图
out = conv(img)
print(out.shape)

# 2、卷积函数进行卷积
kernel = torch.rand((16, 3, 5, 5))  # 卷积核,16个3x5x5的卷积核
bias = torch.rand(16)  # 卷积偏置
out = torch.nn.functional.conv2d(input=img, weight=kernel, bias=bias, stride=1, padding=0)
print(out.shape)
pass

"""
总结：卷积神经网络里面的卷积与图像卷积的区别：
1、pytorch卷积神经网络里的卷积数据只能是张量（4维）
2、卷积神经网络里的卷积有卷积部分和偏置
3、卷积神经网络里的卷积有多个通道，
"""
