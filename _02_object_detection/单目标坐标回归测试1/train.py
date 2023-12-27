"""
1、损失函数
2、优化器
3、训练过程
    训练
    验证
"""
import argparse
import pathlib
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from models.net_resnet18 import net_resnet18
from data.MyData import data_ic
from data.data_xray_毛刺 import data_xray_毛刺


def train(opt):
    print(f"当前程序路径：{os.getcwd()}")
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # 读取数据
    mydata = data_ic('d:/work/files/deeplearn_datasets/test_datasets/单目标回归测试/train')
    #mydata_train = data_xray_毛刺('/mnt/d/work/files/deeplearn_datasets/xray-maoci/train', None)
    datasets_train = DataLoader(mydata, batch_size=5, shuffle=True)

    #mydata_train = data_xray_毛刺('/mnt/d/work/files/deeplearn_datasets/xray-maoci/val', None)
    datasets_val = DataLoader(mydata, batch_size=5, shuffle=True)

    # 训练轮数
    epoch_count = 500
    net = net_resnet18()
    net.to(device)
    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(f"{opt.model_save_path}/logs")

    # 加载预训练模型
    total_loss = 1000
    best_model_path = f'{opt.model_save_path}/weights/best.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        net.load_state_dict(checkpoint['net'])
        total_loss = checkpoint['loss']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"当前损失值：{total_loss}")


    start_epoch = 0
    print(f"训练集的数量:{len(datasets_train)}")
    print(f"验证集的数量:{len(datasets_val)}")
    
    for epoch in range(start_epoch, epoch_count):
        # 训练
        net.train()

        for imgs, labels in datasets_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(out, labels)
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in datasets_val:
                imgs = imgs.to(device)

                labels = labels.to(device)
                out = net(imgs)

                loss = loss_fn(out, labels)
                val_loss += loss


        print(f"epoch:{epoch}, val_loss={val_loss}")

        if val_loss < total_loss:
            total_loss = val_loss
            pathlib.Path(f'{opt.model_save_path}/weights').mkdir(parents=True, exist_ok=True)  # https://zhuanlan.zhihu.com/p/317254621
            
            # 保存训练模型
            state_dict = {'net': net.state_dict(),
                            'optimizer': optimizer.state_dict(),# 不保存优化器权重文件体积非常小，可以上传至github
                            'epoch': epoch,
                            'loss': val_loss}
            torch.save(state_dict, best_model_path)
            print(f"模型参数已保存至：{best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='object_detection/A单目标坐标回归测试1/run/train/weights/best.pth', help='')
    parser.add_argument('--resume', nargs='?', const=True, default=True, help='resume most recent training')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--model_save_path', default='./run/train', help='save to project/name')

    opt = parser.parse_args()
    train(opt)
