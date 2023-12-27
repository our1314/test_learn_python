import torch
from torch import nn
from torch.nn import Module
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

class MyNet(Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#卷积层
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 卷积层
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 卷积层
            nn.ReLU(),
        )
        pass

    def forward(self, x):
        xx = x
        x = self.Conv(x)
        x += xx
        return x

writer = SummaryWriter('log', "graph")
net = MyNet()

x = torch.rand((1,1,3,3))
writer.add_graph(net, x)
pass
