import torch
from torch import nn
from torch.nn import Module, Conv2d, Upsample


class mvtecCAE(Module):
    def __init__(self):
        super(mvtecCAE, self).__init__()

        self.encoded = nn.Sequential(
            torch.nn.Conv2d(3, 32, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(32, 32, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(32, 32, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(64, 128, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(128, 64, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(64, 32, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            torch.nn.Conv2d(32, 1, (7, 7), stride=1, padding=3, bias=False),
            nn.ReLU(),
        )
        
        self.Decode = nn.Sequential(
            Conv2d(1, 32, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            Conv2d(32, 64, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=2),
            Conv2d(64, 128, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=2),
            Conv2d(128, 64, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=2),
            Conv2d(64, 64, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=2),
            Conv2d(64, 32, (3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=2),
            Conv2d(32, 32, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=4),
            Conv2d(32, 32, (4, 4), stride=2, padding=1, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=2),
            Conv2d(32, 32, (7, 7), stride=1, padding=3, bias=False),
            nn.ReLU(),
            Upsample(scale_factor=2),
            Conv2d(32, 3, (7, 7), stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):

        for m in self.encoded:
            x = m(x)
        print(x)
        for i, m in enumerate(self.Decode):
            x = m(x)

        return x


if __name__ == '__main__':
    net = mvtecCAE()
    x = torch.rand((1, 3, 256, 256), dtype=torch.float32)
    y = net(x)
    print(y.shape)
