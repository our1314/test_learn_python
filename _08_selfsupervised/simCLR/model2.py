from typing import Any
import torch
from torch import nn
import torchvision
import math

class Model2(nn.Module):
    def __init__(self, feature_dim=32):
        super(Model2, self).__init__()
        #self.initialize1()
        net = torchvision.models.vgg.vgg11(torchvision.models.VGG11_Weights)
        self.feature = net.features[0:9]#type:nn.Sequential

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640000, 128),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        for m in self.feature:
            x = m.forward(x)
            #print(x.shape)
            pass
        feature = x
        for m in self.classify:
            x = m.forward(x)
        out = x
        return feature, out
    
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize1(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 修改方法一
                m.weight.data = torch.ones(m.weight.data.shape) * 300  # 这样是可以修改的
                # 修改方法二
                #nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
        print('[finishing]:assign weight by inite_weight_1')
if __name__ == '__main__':
    # x = torch.rand((5, 3, 100, 100), dtype=torch.float32)
    # net = torchvision.models.vgg.vgg11(torchvision.models.VGG11_Weights)
    # net = net.features[0:9]
    # print(net)
    # for m in net:
    #     x = m(x)
    #     print(x.shape)
    #     pass

    net = Model2()
    print(net)
    x = torch.rand((5, 3, 200, 200), dtype=torch.float32)
    y = net(x)
    pass