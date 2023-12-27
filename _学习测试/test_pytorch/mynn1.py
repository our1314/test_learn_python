import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.module1 = nn.Sequential(
            Conv2d(3, 64, 3),
            Conv2d(64, 64, 3),
        )

    def forward(self, x):
        x = self.module1(x)
        return x


def train():
    pass


if __name__ == '__main__':
    net = Net()  # 1、定义网络
    loss = nn.CrossEntropyLoss()  # 2、定义损失
    opt = torch.optim.Adam(net.parameters(), lr=0.01)  # 3、定义优化器

    for epoch in range(200):
        # for batch_size in DataLoader:
        #     target_value = ()
        input_data = torch.ones(1, 3, 50, 50)
        output = net(input_data)

        print(output.shape)

        loss_result = loss(output, input_data)

        loss_result.zero_grad()
        loss_result.backward()  # 求梯度（梯度信息存放在网络模型内部）
        opt.step()  # 优化权重
