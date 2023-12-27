import torch
import torchvision
from torch import nn
from torch.nn import Flatten, Linear


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 修改VGG16模型
        vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)

        vgg16.classifier = nn.Linear(25088, 512)
        vgg16.classifier = nn.Linear(512, 4)

        # 将模型赋值到自己的变量
        self.module1 = vgg16
        pass

    def forward(self, x):
        x = self.module1(x)
        pass


def train():
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss = nn.MSELoss()

    for epoch in range(100):
        input_data = torch.ones(1, 3, 48, 48)
        output = net(input_data)
        loss_result = loss(input_data, output)

        loss_result.backward()  # 求梯度
        opt.step()  # 优化网络权重
    pass


if __name__ == '__main__':
    # torch.myutils
    pass
