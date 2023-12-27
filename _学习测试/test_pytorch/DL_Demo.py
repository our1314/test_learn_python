import torch.optim
import torch.nn.functional as F
import torchvision.transforms
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class aa:
    pass


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Layer1 = nn.Sequential(
            nn.Linear(512*612*3, 10),
            nn.ReLU(),
            nn.Linear(10,3)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.Layer1(x)
        x = x.view(-1)
        return x

# 1、准备数据
dataset_train = ImageFolder('D:\\桌面\\有无识别数据集2022.6.20', torchvision.transforms.ToTensor())
dataloader_train = DataLoader(dataset_train, 6, True)

#2、定义网络
net = Net()
#3、定义损失函数
loss_fn = nn.CrossEntropyLoss()
#4、定义优化器
opt = torch.optim.SGD(net.parameters(), lr=0.0001)
#5、设置训练参数


for epoch in range(100):
    for img,target in dataloader_train:
        target = target.float()
        output = net(img)
        loss = loss_fn(output, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        nn.Conv2d()
        pass
