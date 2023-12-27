import cv2
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import data1
from mvtecCAE import mvtecCAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    net = mvtecCAE()
    net.to(device)

    checkpoint = torch.load('best.pth')
    net.load_state_dict(checkpoint['net'])
    print(checkpoint['loss'])

    loss_fn = nn.MSELoss()
    #loss_fn=nn.L1Loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    mydata_train = data1('D:/work/files/deeplearn_datasets/anomalydetection/bottle/train/good')
    datasets_train = DataLoader(mydata_train, batch_size=5, shuffle=True)
    # mydata_val = data1('D:/work/files/deeplearn_datasets/anomalydetection/bottle/val')
    # datasets_val = DataLoader(mydata_val, batch_size=5, shuffle=True)

    last_loss = checkpoint['loss']
    for epoch in range(0, 300):
        print(f"----第{epoch}轮训练开始----")

        # 训练
        net.train()
        total_train_loss = 0

        for imgs, labels in datasets_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(out, labels)
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
    train()
