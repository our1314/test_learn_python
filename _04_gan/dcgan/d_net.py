import torch
from torch import nn


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, True)
        )  # 64, 32, 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )  # 128, 16, 16
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )  # 256, 8, 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )  # 512, 4, 4
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )  # 1, 1, 1
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=9, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )  # 1, 1, 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    # 判别器参数初始化
    def d_weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)


if __name__ == '__main__':
    x = torch.rand((1, 3, 288, 288))
    d_net = D_Net()
    out = d_net(x)
    print(out.shape)
    pass