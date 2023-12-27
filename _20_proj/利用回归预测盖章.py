import torch
from torch.nn import Linear, Module
from torchvision.models.resnet import resnet18, ResNet18_Weights
import os
from torch.utils.data.dataloader import Dataset,DataLoader
import cv2
import numpy as np
import argparse
from torch import nn
import datetime
from random import shuffle
import torchvision
from torchvision.transforms import functional as F

class net_xray(Module):
    def __init__(self, cls_num=1):
        super(net_xray, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.avgpool = nn.Conv2d(512, 1, kernel_size=8)
        self.resnet.fc = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    

train_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomRotation(90)
])

class data_xray_毛刺(Dataset):
    def __init__(self, data_path):
        self.Images = [os.path.join(data_path,'JPEGImages', f) for f in os.listdir(os.path.join(data_path,'JPEGImages'))]  # 列表解析
        self.Labels = [os.path.join(data_path,'Labels', f) for f in os.listdir(os.path.join(data_path,'Labels'))]  # 列表解析

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        # 读取图像
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # type:cv2.Mat
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('dis', image)
        # cv2.waitKey()

        image = image/255.0
        image = cv2.resize(image, (250,250))
        image = np.transpose(image, [2,0,1])
        

        # 读取标签
        f = open(label_path)
        s = f.read().strip()#type:str
        f.close()

        label = torch.tensor([float(s)], dtype=torch.float)
        image = torch.tensor(image, dtype=torch.float)

        image = train_train(image)
        m=torch.max(image)

        F.to_pil_image(image).show()
        return image, label


def train(opt):
    os.makedirs(opt.out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_train = data_xray_毛刺(opt.data_path)
    dataloader_train = DataLoader(datasets_train, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)

    net = net_xray()
    net.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), opt.lr)  # 定义优化器 momentum=0.99
    
    loss_best = 9999
    path_best = os.path.join(opt.out_path, opt.weights)
    if os.path.exists(path_best):
        checkpoint = torch.load(path_best)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        time,epoch,loss = checkpoint['time'],checkpoint['epoch'],checkpoint['loss']
        loss_best = checkpoint['loss']
        print(f"{time}: best.pth, epoch: {epoch}, loss: {loss}")
        pass
    
    for epoch in range(1, opt.epoch):
        # 训练
        net.train()
        loss_train = 0
        for images, labels in dataloader_train:
            images = images.to(device)
            labels = labels.to(device)

            out = net(images)
            loss = loss_fn(input=out, target=labels) #损失函数参数要分input和labels，反了计算值可能是nan 2023.2.24

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            loss_train += loss

        # 打印一轮的训练结果
        loss_train = loss_train / len(dataloader_train.dataset)
        print(f"epoch:{epoch}, loss_train:{loss_train}, lr:{optimizer.param_groups[0]['lr']}")

        # 保存best.pth
        if loss_train < loss_best:
            loss_best = loss_train
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': loss_train.item(),
                          'time': datetime.date.today()}
            torch.save(checkpoint, path_best)
            print(f'已保存:{path_best}')

def predict(opt):
    net = net_xray()
    net.eval()
    path_best = os.path.join(opt.out_path, opt.weights)
    checkpoint = torch.load(path_best)
    net.load_state_dict(checkpoint['net'])

    files = [os.path.join(opt.data_path,'JPEGImages', f) for f in os.listdir(os.path.join(opt.data_path,'JPEGImages'))]  # 列表解析
    shuffle(files)

    for f in files:
        src = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)  # type:cv2.Mat
        h,w,c_ = src.shape
        image = cv2.resize(src, (250,250))
        image = np.transpose(image, [2,0,1])
        image = torch.tensor(image, dtype=torch.float)
        image = torch.unsqueeze(image, 0)

        out = net(image)
        
        dis = cv2.putText(src, str(round(out.item(), 2)), (0,h-1), 0, 1, (0,0,255), 1)
        cv2.imshow('dis', dis)
        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--out_path', default='./run/train', type=str)  # 修改
    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data_path', default='D:/desktop/seal_data(划分之后)/train')  # 修改
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=20, type=int)

    opt = parser.parse_args()

    train(opt)
    #predict(opt)