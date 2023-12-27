import argparse
import os
import pathlib
from datetime import datetime
from random import random
import PIL
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from data import data_seg, trans_train_mask, trans_train_image
from dice_losee import dice_loss
from model import UNet
import math


def train(opt):
    curpath = os.getcwd()
    os.makedirs(opt.out_path, exist_ok=True)
    os.makedirs(f"{opt.out_path}/images", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_train = data_seg('D:/work/files/deeplearn_datasets/test_datasets/xray_real', trans_train_image, trans_train_mask)
    datasets_val = data_seg('D:/work/files/deeplearn_datasets/test_datasets/xray_real')

    dataloader_train = DataLoader(datasets_train, batch_size=4, shuffle=True, num_workers=1, drop_last=True)
    dataloader_val = DataLoader(datasets_val, batch_size=4, shuffle=True, num_workers=1, drop_last=True)

    net = UNet()
    net.to(device)

    loss_fn = nn.BCELoss()
    # loss_fn = dice_loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.99)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), opt.lr)  # 定义优化器 momentum=0.99

    # 学习率更新策略
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epoch)) / 2) * (1 - 0.2) + 0.2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 加载预训练模型
    path_best = f"{opt.out_path}/{opt.weights}"
    if os.path.exists(path_best):
        checkpoint = torch.load(path_best)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"best.pth epoch: {epoch}, loss: {loss}")
    index = 0
    for epoch in range(1, opt.epoch):
        # print(f"----第{epoch}轮训练----")

        # 训练
        net.train()
        loss_train = 0

        # region 设置随机种子
        result_epoch_path = pathlib.Path(f'{opt.out_path}/seed.txt')
        with result_epoch_path.open('w') as fp:
            fp.write(f"{round(random() * 1000000000)}")
        # endregion

        for images, labels in dataloader_train:
            # region 打印训练图像
            if len(os.listdir(f"{opt.out_path}")) < 300:
                for im, la in zip(images, labels):
                    b = im + la
                    a = (b - torch.min(b)) / (torch.max(b) - torch.min(b))
                    c = torchvision.transforms.ToPILImage()(a)  # type: PIL.Image
                    name = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")
                    c.save(f"{opt.out_path}/{name}.png")
                    index += 1
            # endregion

            images = images.to(device)
            labels = labels.to(device)

            out = net(images)
            loss = loss_fn(input=out, target=labels)  # 损失函数参数要分input和labels，反了计算值可能是nan 2023.2.24
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train += loss

        if epoch % 50000000000000 == 0:
            # 验证
            net.eval()
            loss_val = 0
            with torch.no_grad():
                for images, labels in dataloader_val:
                    images = images.to(device)
                    labels = labels.to(device)

                    out = net(images)
                    loss = loss_fn(out, labels)
                    loss_val += loss
                    mean_loss_val = loss_val / len(dataloader_val.dataset)
                print(f"epoch:{epoch}, loss_val:{mean_loss_val}")

        # 打印一轮的训练结果
        mean_loss_train = loss_train / len(dataloader_train.dataset)
        print(f"epoch:{epoch}, loss_train:{mean_loss_train}, lr:{optimizer.param_groups[0]['lr']}")

        # 保存best.pth
        if not os.path.exists(path_best) or (os.path.exists(path_best) and mean_loss_train < checkpoint['loss']):
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': mean_loss_train.item()}
            torch.save(checkpoint, path_best)
            print('已保存best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--out_path', default='./run/train/', type=str)  # 修改
    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data', default='D:/work/files/deeplearn_datasets/test_datasets/xray_real')  # 修改
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--add_graph', default=False, type=bool)
    parser.add_argument('--save_period', default=20, type=int, help='多少轮保存一次，')
    parser.add_argument('--train_img', default=200, type=int, help='保存指定数量的训练图像')

    opt = parser.parse_args()

    train(opt)
