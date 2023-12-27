import random
import time

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(36 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 36 * 6 * 6)
        x = F.relu(self.fc2(F.relu((self.fc1(x)))))
        return x


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.normal_(m.weight)
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.kaiming_normal_(m.weight)
    #         nn.init.constant_(m.weight, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight)


# 取模型中的前四层
# nn.Sequential(*list(net.children())[:4])

# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         nn.init.normal_(m.weight)
#         nn.init.xavier_normal_(m.weight)
#         nn.init.kaiming_normal_(m.weight)#卷积层参数初始化
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m,nn.Linear):
#         nn.init.normal_(m.weight)#全连接层参数初始化

# region 1、初始化数据
# datasets = torchvision.datasets.ImageFolder('C:\\Users\\pc\\Desktop\\有无识别数据集2022.6.20', transform=torchvision.transforms.Compose([
#     torchvision.transforms.Resize((32, 32)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]))
# n = len(datasets)
# n_test = random.sample(range(1, n), int(0.2 * n))  # 按比例取随机数列表 https://blog.csdn.net/TycoonL/article/details/125667592
# dataset_test = torch.myutils.data.Subset(datasets, n_test)
# dataset_train = torch.myutils.data.Subset(datasets, list(set(range(1, n)).difference(set(n_test))))
#
# dataloader_train = DataLoader(dataset_train, 8, shuffle=True)
# dataloader_test = DataLoader(dataset_test, 5, shuffle=True)

datasets_train = torchvision.datasets.cifar.CIFAR10('data', train=True, download=True,
                                                    transform=torchvision.transforms.ToTensor())
dataloader_train = torch.utils.data.DataLoader(datasets_train, batch_size=8, shuffle=True)
datasets_test = torchvision.datasets.cifar.CIFAR10('data', train=False, download=True,
                                                   transform=torchvision.transforms.ToTensor())
dataloader_test = torch.utils.data.DataLoader(datasets_test, batch_size=8, shuffle=True)
# endregion

# region 2、定义网络、损失函数、优化器
net = CNNNet()
loss_fn = nn.CrossEntropyLoss()
optim = optim.SGD(net.parameters(), lr=0.01)

# print(net)
# for aa in net.state_dict():
#     print(net.state_dict()[str(aa)])
#     pass

weight_init(net)
# fc1.weight:tensor([[ 0.0134, -0.0157,  0.0078,  ..., -0.0758,  0.0330, -0.0308],
for aa in net.state_dict():
    print(f'{str(aa)}:{net.state_dict()[str(aa)]}')
    pass

net.to(device)
num_train_loader = len(dataloader_train)
num_test_loader = len(dataloader_test)
for epoch in range(50):
    # 模型训练
    train_loss = 0
    train_acc = 0
    net.train()
    for images, labels in dataloader_train:
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        loss = loss_fn(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
        # print(torch.argmax(output) == labels)
        train_acc += ((torch.argmax(output) == labels).sum()).item()
    print(f'epoch:{epoch}, train, loss:{train_loss / num_train_loader}, acc:{train_acc / num_train_loader}')

    # 模型评估
    eval_loss = 0
    eval_acc = 0
    net.eval()
    for images, labels in dataloader_test:
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        loss = loss_fn(output, labels)
        eval_loss += loss.item()
        eval_acc += ((torch.argmax(output) == labels).sum()).item()
    print(f'epoch:{epoch}, eval, loss:{eval_loss / num_test_loader}, acc:{eval_acc / num_test_loader}')
    pass

    # 保存训练的模型
    model_path = '../OQA2/out'
    if epoch % 5 == 0:
        state_dict = {'net': net.state_dict(),
                      'optimizer': optim.state_dict(),
                      'epoch': epoch}
        p = f'{model_path}/{time.strftime("%Y.%m.%d_%H.%M.%S")}-epoch={epoch}-loss={str(round(train_loss, 5))}-acc={str(round(train_acc, 5))}.pth'
        torch.save(state_dict, p)  # 保存整个模型（包括网络参数和网络结构）
# endregion
