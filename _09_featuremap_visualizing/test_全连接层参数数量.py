'''
假如特征图的通道数为512，全连接层的输出类别数为6，则全连接层为6x512的矩阵，
'''
import torch
from torch import nn

if __name__ == "__main__":
    L1 = nn.Linear(512,2)
    print(L1.weight)
    print(L1.bias)
    print(L1.weight.shape)
    print(L1.bias.shape)

    pass

