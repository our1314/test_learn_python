import math
import torch
import torchvision
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import resnet50, ResNet50_Weights

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        resnet = resnet50()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

if __name__ == "__main__":

    x = torch.rand(1,3,600,600)

    # net = resnet50()
    # y = net(x)
    
    # net = ResNet(Bottleneck, [3, 4, 6, 3])
    # y= net(x)
    # print(y.shape)
    # pass

    net = backbone()
    y = net(x)
    print(y.shape)