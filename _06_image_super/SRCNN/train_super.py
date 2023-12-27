import argparse
import os.path
import pathlib
import sys
from datetime import datetime
import numpy as np
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from image_super.SRCNN.models.net_srcnn import SRCNN
from image_super.SRCNN.utils import utils
from image_super.SRCNN.data import data_test_srcnn


def train(opt):
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练轮数
    epoch_count = opt.epoch
    # 网络
    net = SRCNN(3)  # 加载官方预训练权重

    # 初始化网络权重
    if os.path.exists(opt.weights):
        checkpoint = torch.load(opt.weights)
        net.load_state_dict(checkpoint['net'], strict=False)  # 加载checkpoint的网络权重

    net.to(device)
    loss_fn = nn.L1Loss()
    # loss_fn = nn.MSELoss()  # 定义损失函数
    #loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0.9)

    start_epoch = 0
    if opt.resume:
        start_epoch = checkpoint['epoch']  # 加载checkpoint的优化器epoch
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载checkpoint的优化器

    # 初始化TensorBoard
    writer = SummaryWriter(f"{opt.out_path}/logs")

    # 创建所有文件夹和文件 pathlib.Path
    dirs = ['weights', 'logs', 'img', 'train_fail_img', 'val_fail_img']
    all_dirs = [f"{opt.out_path}/{f}" for f in dirs]  # 列表解析
    [pathlib.Path(f).mkdir(parents=True, exist_ok=True) for f in all_dirs]  # 列表解析
    path_weights, path_logs, path_img, path_train_fail_img, path_val_fail_img = all_dirs

    result_epoch_path = pathlib.Path(f'{opt.out_path}/weights/results.txt')
    result_best_path = pathlib.Path(f'{opt.out_path}/weights/best.txt')
    result_param_path = pathlib.Path(f'{opt.out_path}/weights/param.txt')

    # 保存opt
    with result_param_path.open('w') as fp:
        fp.write(f'{opt}')

    # 绘制网络图
    if opt.add_graph:
        input_size = (1, 3) + opt.datasets
        x = torch.tensor(np.random.randn(input_size), dtype=torch.float)
        x = x.to(device)
        writer.add_graph(net, x)

    # 加载数据集
    data = opt.data
    dataloader_train = DataLoader(data.datasets_train, opt.batch_size, shuffle=True)  # 10
    dataloader_val = DataLoader(data.datasets_val, opt.batch_size, shuffle=True)  # 4

    print(f"训练集的数量：{len(data.datasets_train)}")
    print(f"验证集的数量：{len(data.datasets_val)}")

    cnt = 0
    for epoch in range(start_epoch, epoch_count):
        print(f"----第{epoch}轮训练----")

        # 训练
        net.train()
        acc_train = 0
        loss_train = 0

        for imgs, labels in dataloader_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss

            # region 保存指定数量的训练图像
            img_count = len(os.listdir(path_img))
            if img_count < opt.train_img:
                for i in range(imgs.shape[0]):
                    img = imgs[i, :, :, :]
                    img = torchvision.transforms.ToPILImage()(img)
                    img.save(f'{path_img}/{datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")}.png', 'png')
            # endregion

        # 验证
        net.eval()
        acc_val = 0
        loss_val = 0
        with torch.no_grad():
            for imgs, labels in dataloader_val:
                imgs = imgs.to(device)
                labels = labels.to(device)

                out = net(imgs)
                loss = loss_fn(out, labels)
                loss_val += loss
        '''************************************************分割线***************************************************'''

        # 打印一轮的训练结果
        mean_acc_train = acc_train / len(data.datasets_train)
        mean_loss_train = loss_train
        mean_acc_val = acc_val / len(data.datasets_val)
        mean_loss_val = loss_val

        result_epoch_str = f"epoch:{epoch}, " \
                           f"loss_train:{mean_loss_train}, " \
                           f"loss_val:{mean_loss_val}"

        print(f"{result_epoch_str}\n")

        writer.add_scalar("loss_train", mean_loss_train, epoch)
        writer.add_scalar("loss_val", mean_loss_val, epoch)

        # 保存本轮的训练结果
        with result_epoch_path.open('a') as fp:
            fp.write(f"{result_epoch_str}\n")

        # region 保存模型
        # 保存best
        f = utils.getlastfile(path_weights, ".pth")
        if f is not None:
            checkpoint = torch.load(f)
            # acc_last = checkpoint['acc']
            loss_last = checkpoint['loss']
            if mean_loss_train < loss_last:
                # 保存训练模型
                checkpoint = {'net': net.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'epoch': epoch,
                              # 'acc': mean_acc_train,
                              'loss': mean_loss_train}
                torch.save(checkpoint, f'{opt.out_path}/weights/best.pth')
                print(f"已保存为best.pth\n{result_epoch_str}")
                # 写入best.txt
                with result_best_path.open('w') as fp:
                    fp.write(f'{result_epoch_str}\n')

        # 按周期保存模型
        if epoch % opt.save_period == 0:
            # 创建目录
            # 保存训练模型
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          # 'acc': mean_acc_train,
                          'loss': mean_loss_train}
            torch.save(checkpoint, f'{path_weights}/epoch={epoch}.pth')
            print(f"第{epoch}轮模型参数已保存")
        # endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/train/srcnn/weights/best.pth',  # 修改
                        help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--resume', default=False, type=bool,
                        help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data', default=data_test_srcnn)  # 修改

    parser.add_argument('--epoch', default='40000', type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--out_path', default='run/train/srcnn', type=str)  # 修改
    parser.add_argument('--add_graph', default=False, type=bool)
    parser.add_argument('--save_period', default=4000, type=int, help='多少轮保存一次，')
    parser.add_argument('--train_img', default=200, type=int, help='保存指定数量的训练图像')

    opt = parser.parse_args()
    train(opt)
