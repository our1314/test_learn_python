import argparse
import os.path
import pathlib
from datetime import datetime
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from object_detection.手写yolov1.loss_yolov1 import loss_yolov1
from object_detection.手写yolov1.model.yolov1 import yolov1
from object_detection.手写yolov1.datasets.data_test_yolov1 import data_test_yolov1


def train(opt):
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练轮数
    epoch_count = opt.epoch
    # 网络
    net = yolov1()  # 加载官方预训练权重

    # 初始化网络权重
    if os.path.exists(opt.weights):
        checkpoint = torch.load(opt.weights)
        net.load_state_dict(checkpoint['net'])  # 加载checkpoint的网络权重

    net.to(device)
    loss_fn = loss_yolov1()  # nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    start_epoch = 0
    if opt.resume:
        start_epoch = checkpoint['epoch']  # 加载checkpoint的优化器epoch
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载checkpoint的优化器

    # 初始化TensorBoard
    writer = SummaryWriter(f"{opt.out_path}/logs")
    # 初始化pathlib.Path
    result_epoch_path = pathlib.Path(f'{opt.out_path}/weights/results.txt')
    result_best_path = pathlib.Path(f'{opt.out_path}/weights/best.txt')

    # 绘制网络图
    if opt.add_graph:
        x = torch.tensor(np.random.randn(1, 3, 110, 310), dtype=torch.float)
        x = x.to(device)
        writer.add_graph(net, x)

    # 加载数据集
    data = opt.data  # type:data_test_yolov1
    datasets_train = data("D:/work/files/deeplearn_datasets/test_datasets/test_yolo_xray/train")
    datasets_val = data("D:/work/files/deeplearn_datasets/test_datasets/test_yolo_xray/val")
    dataloader_train = DataLoader(datasets_train, 10, shuffle=True)
    dataloader_val = DataLoader(datasets_val, 4, shuffle=True)

    print(f"训练集的数量：{len(datasets_train)}")
    print(f"验证集的数量：{len(datasets_val)}")
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

            # acc = (out.argmax(1) == labels).sum()
            # acc_train += acc
            loss_train += loss

            # region 保存指定数量的训练图像
            path = f'{opt.out_path}/img'
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            img_count = len(os.listdir(path))
            if img_count < opt.train_img:
                cnt += 1
                path = f'{opt.out_path}/img'
                if os.path.exists(path) is not True:
                    os.makedirs(path)

                for i in range(imgs.shape[0]):
                    img = imgs[i, :, :, :]
                    img = torchvision.transforms.ToPILImage()(img)
                    img.save(f'{opt.out_path}/img/{datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")}.png', 'png')
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
                # acc = (out.argmax(1) == labels).sum()
                # acc_val += acc
                loss_val += loss

        '''************************************************分割线***************************************************'''
        # 打印一轮的训练结果
        # mean_acc_train = acc_train / len(data_xray.datasets_train)
        mean_loss_train = loss_train
        # mean_acc_val = acc_val / len(data_xray.datasets_val)
        mean_loss_val = loss_val

        result_epoch_str = f"epoch:{epoch}, " \
                           f"loss_train:{mean_loss_train}, " \
                           f"loss_val:{mean_loss_val}"

        print(f"{result_epoch_str}\n")

        writer.add_scalar("loss_train", mean_loss_train, epoch)
        writer.add_scalar("loss_val", mean_loss_val, epoch)
        checkpoint = {'net': net.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch,
                      'loss_train': loss_train,
                      'loss_val': loss_val}
        pathlib.Path(f'{opt.out_path}/weights').mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, f'{opt.out_path}/weights/best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', default='run/train/exp/weights/best.pth',
                        help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--resume', default=False, type=bool,
                        help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data', default=data_test_yolov1)
    parser.add_argument('--num_class', default=2, type=int)

    parser.add_argument('--epoch', default='300', type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--out_path', default='run/train/exp', type=str)
    parser.add_argument('--add_graph', default=False, type=bool)
    parser.add_argument('--save_period', default=20, type=int, help='多少轮保存一次，')
    parser.add_argument('--train_img', default=200, type=int, help='保存指定数量的训练图像')

    opt = parser.parse_args()
    train(opt)
