"""
测试基于坐标回归来进行目标定位，直接进行坐标回归，不考虑长宽
"""
import datetime
import os
import torchvision
import cv2
import numpy as np
import torch
from cv2.mat_wrapper import Mat
from torch import nn
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        resnet18 = torchvision.models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(512, 2)

        self.Model1 = resnet18

    def forward(self, x):
        x = self.Model1(x)
        return x


class Data(Dataset):
    def __init__(self, data_path):
        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "Images")
        labels_path = os.path.join(data_path, "Labels")

        for filename in os.listdir(images_path):
            self.Images.append(os.path.join(images_path, filename))

        for filename in os.listdir(labels_path):
            self.Labels.append(os.path.join(labels_path, filename))
        pass

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        # img = torch.FloatTensor(img)
        img = torchvision.transforms.ToTensor()(img)
        # img = torchvision.transforms.Resize(300)(img)

        # img = torch.float(img)
        img /= 255.
        # img = torch.permute(img, (2, 0, 1))  # 交换维度

        pos = []
        with open(label_path, 'r', encoding='utf-8') as f:
            list = f.readlines()
            data_list = list[0].split(' ')
            x = float(data_list[1])
            y = float(data_list[2])
            pos = torch.tensor([x, y])

        return img, pos


class Train:
    def __init__(self):
        data = Data('C:\\work\\files\\dataset\\坐标回归测试')
        self.train_dataloader = DataLoader(data, batch_size=32, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Net = Net().to(self.device)
        self.Opt = torch.optim.Adam(self.Net.parameters(), lr=0.0001)
        self.Loss = nn.MSELoss()
        pass

    def __call__(self, *args, **kwargs):
        for epoch in range(300):
            l = 0.
            for i, (img, label) in enumerate(self.train_dataloader):  # 读数据
                img, label = img.to(self.device), label.to(self.device)
                pos_output = self.Net(img)  # 前向传播
                pos_loss = self.Loss(pos_output, label)  # 位置损失

                self.Opt.zero_grad()  # 梯度清零(必须在求梯度之前)
                pos_loss.backward()  # 计算梯度
                self.Opt.step()  # 优化权重
                # if epoch % 10 == 0:
                l += pos_loss.item()
            print(f'train_loss  epoch = {epoch}, loss = {l}')
            if epoch % 10 == 0:
                date_time = str(datetime.datetime.now())
                torch.save(self.Net.state_dict(), f'{epoch}-{round(l, 5)}.pth')


class Test:
    def __init__(self):
        self.net = Net()

    def __call__(self, ):
        self.net.load_state_dict()


if __name__ == '__main__':
    # train = Train()
    # train()
    import our1314



    net = Net()
    net.load_state_dict(torch.load('290-0.00081.pth'))
    net.eval()

    data = Data('C:\\work\\files\\dataset\\坐标回归测试')
    train_dataloader = DataLoader(data, batch_size=1, shuffle=False)

    shape = torch.tensor([[306.,256.]])
    for i, (img, label) in enumerate(train_dataloader):
        loc = net.forward(img)
        loc_real = torch.mul(shape, loc)

        dis = (img * 255).numpy()

    net = Net()  # .to('cuda')
    input_data = torch.ones(1, 3, 512, 612)

    output_data = net(input_data)
    print(output_data)
    pass
    # d = Data('D:\\桌面\\图像采集\\OQA IC检测测试图像\\OQA IC检测\\out5')
    # img, x, y = d[0]
    #
    # img = torch.permute(img, (1, 2, 0))
    # img = img.numpy()
    # cv2.imshow('1', cv2.pyrDown(img))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # b = DataLoader(d, batch_size=10, shuffle=True)
    # pass
