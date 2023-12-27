import argparse
import os
import cv2
import torch
import data
from torchvision.transforms import transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from model1 import Model1
from model2 import Model2
from data import ChouJianJi
from loss import Loss_fn
import sys
sys.path.append("../../")
import myutils.myutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(opt):
    net = Model1()
    net.to(device)
    
    loss_fn = Loss_fn(opt.temperature, opt.batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-6)

    if os.path.exists('best.pth'):
        checkpoint = torch.load('best.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(checkpoint['loss'])

    mydata_train = ChouJianJi('D:/work/proj/抽检机/program/ChouJianJi/data/ic', data.train_transform)
    datasets_train = DataLoader(mydata_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    mydata_val = ChouJianJi('D:/work/proj/抽检机/program/ChouJianJi/data/ic', data.test_transform)
    datasets_val = DataLoader(mydata_val, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    last_loss = checkpoint['loss'] if os.path.exists('best.pth') else 1000.0
    for epoch in range(0, 3000):
        print(f"----第{epoch}轮训练开始----")

        # 训练
        net.train()
        total_train_loss = 0

        for img1, img2 in datasets_train:
            img1 = img1.to(device)
            img2 = img2.to(device)

            fea1,out1 = net(img1)
            fea2,out2 = net(img2)
            loss = loss_fn(out1, out2)
            loss /= opt.batch_size
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss

        # # 验证
        # net.eval()
        # total_val_loss = 0
        # total_val_accuracy = 0
        # with torch.no_grad():
        #     for imgs, labels in datasets_val:
        #         imgs = imgs.to(device)
        #
        #         labels = labels.to(device)
        #         out = net(imgs)
        #
        #         loss = loss_fn(out, labels)
        #         total_val_loss += loss
        #
        #         acc = loss
        #         total_val_accuracy += acc

        #保存best训练模型
        train_loss = total_train_loss
        val_loss = 0  # total_val_loss

        print(f"epoch:{epoch}, train_loss={train_loss}, val_loss={val_loss}")

        if train_loss < last_loss:
            last_loss = train_loss
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),  # 不保存优化器权重文件体积非常小，可以上传至github
                          'epoch': epoch,
                          'loss': train_loss}
            torch.save(checkpoint, f'best.pth')
            print(f"\t当前模型参数已保存为best")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    opt = parser.parse_args()
    train(opt)
