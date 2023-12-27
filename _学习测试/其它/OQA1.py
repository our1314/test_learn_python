import random
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


nn.Loss
class CNNNet123(nn.Module):
    def __init__(self):
        super(CNNNet123, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(36*6*6, 100)
        self.fc2 = nn.Linear(100, 3)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 36*6*6)
        x = F.relu(self.fc2(F.relu((self.fc1(x)))))
        return x

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        self.resnet.fc.out_features = 3

    def forward(self, x):
        x = self.resnet(x)
        return x

#取模型中的前四层
# nn.Sequential(*list(net.children())[:4])

# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         nn.init.normal_(m.weight)
#         nn.init.xavier_normal_(m.weight)
#         nn.init.kaiming_normal_(m.weight)#卷积层参数初始化
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m,nn.Linear):
#         nn.init.normal_(m.weight)#全连接层参数初始化



#


#region 1、初始化数据
datasets = torchvision.datasets.ImageFolder('D:\\桌面\\有无识别数据集2022.6.20', transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
]))
n = len(datasets)
n_test = random.sample(range(1, n), int(0.2 * n))  #按比例取随机数列表 https://blog.csdn.net/TycoonL/article/details/125667592
dataset_test = torch.utils.data.Subset(datasets, n_test)
dataset_train = torch.utils.data.Subset(datasets, list(set(range(1, n)).difference(set(n_test))))

dataloader_train = DataLoader(dataset_train, 8, shuffle=True)
dataloader_test = DataLoader(dataset_test, 5, shuffle=True)
#endregion

#region 2、定义网络、损失函数、优化器
net = CNNNet()
loss_fn = nn.CrossEntropyLoss()
optim = optim.SGD(net.parameters(), lr=0.01)
# x = torch.randn(1, 3, 512, 612, device=device)
# out = net(x)

net.to(device)
net.train()

for epoch in range(50):
    train_loss = 0
    train_acc = 0

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
        train_acc += ((torch.argmax(output) == labels).sum()/images.shape[0]).item()
    print(f'epoch:{epoch}, loss:{train_loss / len(dataloader_train)}, acc:{train_acc / len(dataloader_train)}')
    print('123')
#endregion