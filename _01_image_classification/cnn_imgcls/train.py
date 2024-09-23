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
from data import data_oqa_agl, data_oqa_chr, data_cleaner, data_wide_resnet
    # , data_xray_sot23, data_xray_sc88, data_xray_sc70, data_xray_sc89, \
    # data_xray_sod123, data_xray_sod323, data_xray_sot23_juanpan, data_xray_sod523, data_xray_sod723, data_xray_sot25, \
    # data_xray_sot26, data_xray_sot23e, ,  data_cleaner, data_xray_allone, data_xray_maoci


from model import net_xray, wide_resnet
import sys
sys.path.append("D:/work/program/Python/DeepLearning/test_learn_python")
from our1314 import work


def train(opt):
    # 定义设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 训练轮数
    epoch_count = opt.epoch
    # 网络
    net = net_xray(True, opt.data.class_num)  # 加载官方预训练权重

    # 初始化网络权重
    if os.path.exists(opt.pretrain):
        checkpoint = torch.load(opt.pretrain)
        net.load_state_dict(checkpoint['net'], strict=False)  # 加载checkpoint的网络权重

    net.to(device)
    loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)  # 定义优化器 momentum=0.99
    # optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

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

    acc_best = 0 #checkpoint['acc']
    loss_best = 999 #checkpoint['loss']
    for epoch in range(start_epoch, epoch_count):
        print(f"----第{epoch}轮训练----")

        # 训练
        net.train()
        acc_train,loss_train,acc_train_cnt = 0,0,0

        for imgs, labels in dataloader_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (out.argmax(1) == labels).sum()
            acc_train_cnt += acc
            loss_train += loss

            # region 保存指定数量的训练图像
            img_count = len(os.listdir(path_img))
            if img_count < opt.train_img:
                for i in range(imgs.shape[0]):
                    img = imgs[i, :, :, :]
                    img = torchvision.transforms.ToPILImage()(img)
                    img.save(f'{path_img}/{datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")}.png', 'png')
            # endregion

            # region 保存训练失败的图像
            img_count = len(os.listdir(path_train_fail_img))
            if img_count < 30:
                out_arg = out.argmax(1)
                check_result = out_arg == labels

                #print(oo)
                for i in range(check_result.shape[0]):
                    f = check_result[i]
                    if f == False:
                        img = imgs[i, :, :, :]
                        img = torchvision.transforms.ToPILImage()(img)
                        img.save(f'{path_train_fail_img}/{datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")}_out_{out_arg[i]}_label_{labels[i]}.png')

            # endregion

        # 验证
        net.eval()
        acc_val,loss_val,acc_val_cnt = 0,0,0

        with torch.no_grad():
            for imgs, labels in dataloader_val:
                imgs = imgs.to(device)
                labels = labels.to(device)

                out = net(imgs)
                loss = loss_fn(out, labels)
                acc = (out.argmax(1) == labels).sum()
                acc_val_cnt += acc
                loss_val += loss

                # region 保存验证失败的图像
                img_count = len(os.listdir(path_val_fail_img))
                if img_count < 30:
                    out_arg = out.argmax(1)
                    check_result = out_arg == labels
                    for i in range(check_result.shape[0]):
                        f = check_result[i]
                        if f == False:
                            img = imgs[i, :, :, :]
                            img = torchvision.transforms.ToPILImage()(img)
                            img.save(f'{path_val_fail_img}/{datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")}_out_{out_arg[i]}_label_{labels[i]}.png')
                # endregion

        '''************************************************分割线***************************************************'''

        # 打印一轮的训练结果
        acc_train = acc_train_cnt / len(data.datasets_train)
        loss_train = loss_train / len(data.datasets_train)
        acc_val = acc_val_cnt / len(data.datasets_val)
        loss_val = loss_val / len(data.datasets_val)

        result_epoch_str = f"epoch:{epoch}, " \
                           f"lr:{optimizer.state_dict()['param_groups'][0]['lr']} " \
                           f"acc_train:{acc_train}({acc_train_cnt}/{len(data.datasets_train)}) " \
                           f"loss_train:{loss_train}, " \
                           f"acc_val:{acc_val}({acc_val_cnt}/{len(data.datasets_val)}) " \
                           f"loss_val:{loss_val}"

        print(f"{result_epoch_str}\n")

        writer.add_scalar("acc_train", acc_train, epoch)
        writer.add_scalar("loss_train", loss_train, epoch)
        writer.add_scalar("acc_val", acc_val, epoch)
        writer.add_scalar("loss_val", loss_val, epoch)

        # 保存本轮的训练结果
        with result_epoch_path.open('a') as fp:
            fp.write(f"{result_epoch_str}\n")

        # 保存权重         
        if acc_val >= acc_best and loss_val < loss_best:
            acc_best = acc_val
            loss_best = loss_val

            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'acc': acc_train,
                          'loss': loss_train}
            torch.save(checkpoint, os.path.join(opt.out_path,'weights',opt.weights))
            print(f'已保存:{opt.weights}')
            # 写入best.txt
            with result_best_path.open('w') as fp:
                fp.write(f'{result_epoch_str}\n')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pretrain', default='./run/train/chr/weights/best.pth', help='指定权重文件，未指定则使用官方权重！')  # 修改
#     parser.add_argument('--out_path', default='./run/train/chr/weights/best_2024-8-24.pth', type=str)  # 修改
#     parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')
#     parser.add_argument('--data', default=data_oqa_chr)#修改

#     parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
#     parser.add_argument('--epoch', default=5000, type=int)
#     parser.add_argument('--lr', default=0.01, type=float)
#     parser.add_argument('--batch_size', default=10, type=int)
#     parser.add_argument('--add_graph', default=False, type=bool)
#     parser.add_argument('--save_period', default=20, type=int, help='多少轮保存一次，')
#     parser.add_argument('--train_img', default=100, type=int, help='保存指定数量的训练图像')

#     opt = parser.parse_args()
#     train(opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default='./run/train/agl/weights/best.pth', help='指定权重文件，未指定则使用官方权重！')  # 修改
    parser.add_argument('--out_path', default='./run/train/agl/weights/best_oqa_2024-8-24.pth', type=str)  # 修改
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--data', default=data_oqa_agl)#修改

    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--add_graph', default=False, type=bool)
    parser.add_argument('--save_period', default=20, type=int, help='多少轮保存一次，')
    parser.add_argument('--train_img', default=100, type=int, help='保存指定数量的训练图像')

    opt = parser.parse_args()
    train(opt)