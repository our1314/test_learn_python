import os
import numpy
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import utils123.utils


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class data_srcnn(Dataset):
    def __init__(self, data_path, num_channels=3):
        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "labels")

        self.Images = [images_path + '/' + f for f in os.listdir(images_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.Labels = [labels_path + '/' + f for f in os.listdir(labels_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        image = Image.open(image_path)
        label = Image.open(label_path)
        # label.show()

        image = image.convert("RGB")
        label = label.convert("RGB")
        # label.show()

        image = torchvision.transforms.ToTensor()(image)
        label = torchvision.transforms.ToTensor()(label)

        return image, label


datasets_train = data_srcnn("D:/work/files/deeplearn_datasets/test_datasets/xray_super")
datasets_val = data_srcnn("D:/work/files/deeplearn_datasets/test_datasets/xray_super")


class loss1(nn.Module):
    def __init__(self):
        super(loss1, self).__init__()
        self.device = device
        self.loss = nn.MSELoss()

    def forward(self, x1, x2):
        ones = torch.ones_like(x1)
        zeros = torch.zeros_like(x1)

        mask = torch.where(x1 == 0, x1, ones)  # 只有线条区域为0
        xx1 = ones - mask
        xx2 = torch.where(mask == 0, x2, zeros)

        m1 = torch.max(xx2)
        m2 = torch.min(xx2)

        # img1 = torchvision.transforms.ToPILImage()(xx1[0])
        # img1.show()
        #
        # img2 = torchvision.transforms.ToPILImage()(x2[0])
        # img2.show()
        # img2 = torchvision.transforms.ToPILImage()(xx2[0])
        # img2.show()

        #loss = torch.abs((xx1 - xx2).sum())
        loss = self.loss(xx1, xx2)

        # m1 = torch.max(mask)
        # m2 = torch.min(mask)

        # loss = x1 - x2
        # # loss = torch.sum(loss)
        # # loss = 1/(1+torch.exp(-x))
        # loss = torch.mean(loss)
        # # loss = torch.sigmoid(loss)
        #
        # mask = x1 = 0

        # 将x1中为0的位置与x2的对应位置计算损失
        # 保证其它位置计算时损失不变，只有这个位置才会变化

        return loss

    def compute_loss(pred_tensor, label_tensor):
        """
        计算无目标的损失、置信度损失、BBOX损失、分类损失
        1、无目标的网格，直接计算其置信度损失（label置信度为0）
        2、有目标的网格，需要计算其：
                                置信度损失
                                xy损失
                                wh损失
                                cls损失
        """
        pass


if __name__ == '__main__':

    a = numpy.asarray([[0, 0, 0], [128, 128, 128], [255, 255, 255]], numpy.uint8)
    a = utils.utils.mat2pil(a)
    b = torchvision.transforms.ToTensor()(a)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader_train = DataLoader(datasets_train, 2, shuffle=True)  # 10
    dataloader_val = DataLoader(datasets_val, 2, shuffle=True)  # 4
    net = SRCNN(3)  # 加载官方预训练权重
    net.to(device)

    loss_fn = loss1()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # 定义优化器 momentum=0.99
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)  # 定义优化器 momentum=0.99
    cnt = 0
    for epoch in range(0, 1000):
        print(f"----第{epoch}轮训练----")

        # 训练
        net.train()
        acc_train = 0
        loss_train = 0

        for imgs, labels in dataloader_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(labels, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss

        # 打印一轮的训练结果
        mean_loss_train = loss_train

        result_epoch_str = f"epoch:{epoch}, " \
                           f"loss_train:{mean_loss_train}"

        print(f"{result_epoch_str}\n")

    checkpoint = {'net': net.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch,
                  'loss': mean_loss_train}
    torch.save(checkpoint, f'./epoch={epoch}.pth')
