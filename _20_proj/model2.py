import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Upsample
from torchvision.models import resnet50, resnet18, vgg16
import torchvision

class Model2(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model2, self).__init__()

        self.vgg = vgg16(torchvision.models.VGG16_Weights)

    def forward(self, x):
        sz = 100
        last_shape = 0
        for m in self.vgg.features:
            x = m(x)
            print(x.shape)
            if(last_shape==sz and x.shape[2]!=sz):
                return x
            last_shape=x.shape[2]
            
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


if __name__ == '__main__':
    net = Model2()
    print(net)
    x = torch.rand((5, 3, 100, 100), dtype=torch.float32)
    y = net(x)
    pass
