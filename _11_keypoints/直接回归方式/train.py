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
import torchvision
from data_keypoints import data_keypoints
from net import net_resnet18, net_resnet50

def train(opt):
    print(f"当前程序路径：{os.getcwd()}")
    pathlib.Path(f"{opt.run_path}/logs").mkdir(parents=True, exist_ok=True)
    # 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读取数据
    mydata = data_keypoints('D:/work/files/deeplearn_datasets/choujianji/roi-keypoint')
    datasets_train = DataLoader(mydata, batch_size=5, shuffle=True)
    datasets_val = DataLoader(mydata, batch_size=5, shuffle=True)

    # 训练轮数
    epoch_count = 500000
    net = net_resnet50()
    net.to(device)
    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.973)
    #writer = SummaryWriter(f"{opt.run_path}/logs")

    # 加载预训练模型
    total_loss = torch.tensor(1000.0).to(device)
    best_model_path = f'{opt.run_path}/weights/best.pth'
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
        train_loss = 0
        for imgs, labels in datasets_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(out, labels)
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        # # 验证
        # net.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for imgs, labels in datasets_val:
        #         imgs = imgs.to(device)
        #         labels = labels.to(device)

        #         out = net(imgs)
        #         loss = loss_fn(out, labels)
        #         val_loss += loss

        print(f"epoch:{epoch}, val_loss={train_loss}")

        if train_loss < total_loss:
            total_loss = train_loss
            pathlib.Path(f'{opt.run_path}/weights').mkdir(parents=True, exist_ok=True)  # https://zhuanlan.zhihu.com/p/317254621
            
            # 保存训练模型
            state_dict = {'net': net.state_dict(),
                            'optimizer': optimizer.state_dict(),# 不保存优化器权重文件体积非常小，可以上传至github
                            'epoch': epoch,
                            'loss': train_loss}
            torch.save(state_dict, best_model_path)
            print(f"模型参数已保存至：{best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', nargs='?', default='./run_resnet50', help='')
    parser.add_argument('--resume', nargs='?', const=True, default=True, help='resume most recent training')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')

    opt = parser.parse_args()
    train(opt)
