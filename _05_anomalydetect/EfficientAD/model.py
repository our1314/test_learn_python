import torch
from torch import nn


class PDN_small(nn.Module):
    def __init__(self, out_channels=384, padding=False):
        super(PDN_small, self).__init__()

        pad_mult = 1 if padding else 0

        self.pdn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1 * pad_mult),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
        )

    def forward(self, x):
        x = self.pdn(x)
        return x
    

class PDN_medium(nn.Module):
    def __init__(self, out_channels=384, padding=False):
        super(PDN_medium, self).__init__()

        pad_mult = 1 if padding else 0

        self.pdn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1 * pad_mult),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.pdn(x)
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, out_channels=384):
        super(AutoEncoder, self).__init__()

        self.ae = nn.Sequential(
            # encoder
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),

            # decoder
            nn.Upsample(size=3, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=8, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=15, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=32, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=63, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=127, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=56, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        x = self.ae(x)
        return x


class Teacher(PDN_medium):
    pass


class Student(PDN_medium):
    pass


if __name__ == "__main__":
    x = torch.rand([1,3,256,256])
    t = Teacher()
    x = t(x)
    print(x.shape)
    pass