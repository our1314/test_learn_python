import torch
from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=5 // 2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv5 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


if __name__ == '__main__':
    img = torch.rand((1, 3, 256, 697))
    net = SRCNN(3)
    d = net(img)
    pass
