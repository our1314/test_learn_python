import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
#from data_抽检机 import data_seg, transform1, transform2, transform_val
from data_空洞检测 import data_seg, transform1, transform2, transform_val
#from data_切割道检测 import data_seg, transform1, transform2, transform_val
from model import deeplabv3,UNet,DeepLabV3Plus
import datetime
import tqdm


def train(opt):
    os.makedirs(opt.out_path, exist_ok=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    datasets_train = data_seg(opt.data_path_train, transform1, transform2)
    datasets_val = data_seg(opt.data_path_val, transform_val)

    dataloader_train = DataLoader(datasets_train, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=False)
    dataloader_val = DataLoader(datasets_val, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=False)

    net = deeplabv3()#DeepLabV3Plus(n_classes=2,n_blocks=[3, 4, 23, 3],atrous_rates=[6, 12, 18],multi_grids=[1, 2, 4],output_stride=16) #deeplabv3()  # deeplabv3() UNet() DeepLabV3_2()
    net.to(device)

    #loss_fn = nn.BCELoss(reduction='mean')
    #https://blog.csdn.net/weixin_41321482/article/details/110395017
    #ignore_index用于忽略ground-truth中某些不需要参与计算的类。假设有两类{0:背景，1：前景}，若想在计算交叉熵时忽略背景(0)类，则可令ignore_index=0（同理忽略前景计算可设ignore_index=1）。
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), opt.lr)  # 定义优化器 momentum=0.99

    # 学习率更新策略
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch/20, eta_min=1e-5)


    # 加载预训练模型
    loss_best = 9999
    if os.path.exists(opt.pretrain):
        try:
            checkpoint = torch.load(opt.pretrain)
            net.load_state_dict(checkpoint['net'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            time,epoch,loss = checkpoint['time'],checkpoint['epoch'],checkpoint['loss']
            loss_best = checkpoint['loss']
            print(f"加载权重: {opt.pretrain}, {time}: epoch: {epoch}, best loss: {loss}")
        except:
            print("加载权重出现异常！")
    else:
        print("未找到预训练权重文件！")
    
    for epoch in range(1, opt.epoch):
        # 训练
        net.train()
        loss_train = 0
        for images, labels in tqdm.tqdm(dataloader_train, "train"):
            images = images.to(device)
            labels = labels.to(device)

            out = net(images)
            loss = loss_fn(input=out, target=labels) #损失函数参数要分input和labels，反了计算值可能是nan 2023.2.24
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()

        scheduler.step()

        # 验证
        net.eval()
        loss_val = 0
        with torch.no_grad():
            for images, labels in tqdm.tqdm(dataloader_val, "val"):
                images = images.to(device)
                labels = labels.to(device)
                out = net(images)
                loss = loss_fn(input=out, target=labels) #损失函数参数要分input和labels，反了计算值可能是nan 2023.2.24
                loss_val += loss.item()

        # 打印一轮的训练结果
        loss_train = loss_train / len(datasets_train)
        loss_val = loss_val / len(datasets_val)
        print(f"epoch:{epoch}, loss_train:{round(loss_train, 6)}, loss_val:{round(loss_val, 6)}, lr:{optimizer.param_groups[0]['lr']}")

        # 保存权重
        if loss_train < loss_best:
            loss_best = loss_train
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': loss_best,
                          'time': datetime.date.today()}
            torch.save(checkpoint, os.path.join(opt.out_path,opt.weights))
            print(f'已保存:{opt.weights}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default='./run/train/best_test.pth', help='指定权重文件，未指定则使用官方权重！')  # 修改
    parser.add_argument('--out_path', default='./run/train', type=str)  # 修改
    parser.add_argument('--weights', default='best_test.pth', help='指定权重文件，未指定则使用官方权重！')

    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data_path_train', default='D:/work/files/deeplearn_datasets/xray空洞检测/生成数据Power/train')
    parser.add_argument('--data_path_val', default='D:/work/files/deeplearn_datasets/xray空洞检测/生成数据Power/val')
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=14, type=int)

    opt = parser.parse_args()

    train(opt)
