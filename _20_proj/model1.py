import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Upsample
from torchvision.models import resnet50, resnet18


class Model1(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model1, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 128, bias=False), nn.BatchNorm1d(128),
                               nn.ReLU(inplace=True), nn.Linear(128, feature_dim, bias=True))
        




    def forward(self, x):
        for m in self.f:
            x = m(x)
            if(x.shape[2]==25):
                return x
            #print(x.shape)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


if __name__ == '__main__':
    net = Model1()
    print(net)
    x = torch.rand((5, 3, 100, 100), dtype=torch.float32)
    y = net(x)
    pass
