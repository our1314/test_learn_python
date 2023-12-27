import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Upsample
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        #x = self.f(x)
        for m in self.f:
            x = m(x)
            # print(x.name)
            # print(x.shape)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


if __name__ == '__main__':
    net = Model()
    print(net)
    x = torch.rand((5, 3, 100, 100), dtype=torch.float32)
    y = net(x)
    pass
