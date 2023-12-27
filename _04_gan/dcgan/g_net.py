import torch
from torch import nn


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )  # 512, 4, 4
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )  # 256, 8, 8
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )  # 128, 16, 16
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )  # 128, 32, 32
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 3, 1, bias=False),
            nn.Tanh()
        )  # 3, 96, 96
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )  # 3, 184, 184

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    # 生成器参数初始化
    def g_weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)


if __name__ == '__main__':
    z = torch.randn(1, 128, 1, 1)
    g_net = G_Net()
    out = g_net(z)
    print(out.shape)
    pass
