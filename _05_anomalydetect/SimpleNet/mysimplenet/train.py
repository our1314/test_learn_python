import argparse
import os
from data import DatasetSplit,IMAGENET_MEAN,IMAGENET_STD
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import CJJDataset
#from model.model import SimpleNet
from simplenet import SimpleNet
import datetime 
import random
import numpy as np
import cv2


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.
    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train(opt):
    os.makedirs(opt.out_path, exist_ok=True)

    fix_seeds(0)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    datasets_train = CJJDataset(opt.data_path) #MVTecDataset(opt.data_path, "pill")
    #datasets_val = MVTecDataset(opt.data_path_val, "pill")

    dataloader_train = DataLoader(datasets_train, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)
    #dataloader_val = DataLoader(datasets_val, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)

    net = SimpleNet()
    net.to(device)

    #loss_fn = nn.BCELoss(reduction='mean')
    #loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), opt.lr)  # 定义优化器 momentum=0.99

    # 学习率更新策略
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)


    # 加载预训练模型
    loss_best = 9999
    if os.path.exists(opt.pretrain):
        checkpoint = torch.load(opt.pretrain)
        # net.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        time,epoch,loss = checkpoint['time'],checkpoint['epoch'],checkpoint['loss']
        #loss_best = checkpoint['loss']
        print(f"加载权重: {opt.pretrain}, {time}: epoch: {epoch}, loss: {loss}")
    
    for epoch in range(1, opt.epoch):
        # 训练
        #net.train()
        # net.backbone.eval()
        # net.project.train()
        # net.discriminator.train()

        loss_train = 0

        all_loss=[]
        all_p_true=[]
        all_p_fake=[]
        for data in dataloader_train:
            images = data["image"]
            images = images.to(device)

            loss,p_true,p_fake = net(images)
            all_loss.append(loss)
            all_p_true.append(p_true)
            all_p_fake.append(p_fake)

        #scheduler.step()

        all_loss = sum(all_loss) / len(dataloader_train)
        all_p_true = sum(all_p_true) / len(dataloader_train)
        all_p_fake = sum(all_p_fake) / len(dataloader_train)

        # 验证
        # net.eval()
        # loss_val = 0
        # with torch.no_grad():
        #     for images, labels in dataloader_val:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         out = net(images)
        #         loss = loss_fn(input=out, target=labels) #损失函数参数要分input和labels，反了计算值可能是nan 2023.2.24
        #         loss_val += loss.item()

        # 打印一轮的训练结果
        loss_train = loss_train / len(dataloader_train)
        #loss_val = loss_val / len(dataloader_val)
        print(f"epoch:{epoch}, loss_train:{round(all_loss, 6)}, p_true:{round(all_p_true, 6)}, p_fake:{round(all_p_fake, 6)}, lr:{optimizer.param_groups[0]['lr']}")

        loss_train = all_loss
        # 保存权重
        if loss_train < loss_best:
            loss_best = loss_train
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': loss_train,
                          'time': datetime.date.today()}
            torch.save(checkpoint, os.path.join(opt.out_path,opt.weights))
            print(f'已保存:{opt.weights}')

def predict(opt):
    device = torch.device("cpu")
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = SimpleNet()
    checkpoint = torch.load(opt.pretrain)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net = net.cpu()
    #datasets_train = MVTecDataset(opt.data_path_train, "pill")
    datasets_test = CJJDataset(opt.data_path, split=DatasetSplit.TEST)
    i=0
    for data in datasets_test:
        images = data['image']#type:torch.Tensor
        #images.to(device)
        images = images.unsqueeze(0)
        masks = net.predict(images)
        #continue
        max_value = np.round(np.max(masks),2)
        min_value = np.round(np.min(masks),2)
        print("\nmax=",max_value,"min=",min_value)
        
    
        img = images[0]
        img = img.cpu().numpy()
        img = img.transpose([1,2,0])
        img = img*IMAGENET_STD + IMAGENET_MEAN
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("dis", img)
        cv2.waitKey()
        continue

        temp = masks[0]
        _,thr = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY)
        thr = thr.astype("uint8")
        temp = 1/(1+np.exp(-temp))#sigmoid

        #region 将mask转换为热力图
        temp = np.uint8(temp*255)
        img = np.uint8(img*255)
        temp = cv2.applyColorMap(temp,cv2.COLORMAP_JET)
        #endregion

        contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        h,w,c = img.shape
        dis = cv2.hconcat([img,temp])
        cv2.drawContours(dis, contours, -1, (0,0,255),3)
        cv2.putText(dis, data['filename'], (0,18), cv2.FONT_ITALIC, 0.7, (0,0,255), 1)
        cv2.putText(dis, f"max={str(max_value)},min={str(min_value)}", (0,h-2), cv2.FONT_ITALIC, 0.8, (0,0,255), 1)
        cv2.imshow("dis", dis)
        cv2.waitKey()
        

def predict1(opt):
    device = torch.device("cpu")
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = SimpleNet()
    checkpoint = torch.load(opt.pretrain)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net = net.cpu()
    #datasets_train = MVTecDataset(opt.data_path_train, "pill")
    datasets_test = CJJDataset(opt.data_path, split=DatasetSplit.TEST)
    i=0
    for data in datasets_test:
        images = data['image']#type:torch.Tensor
        #images.to(device)
        images = images.unsqueeze(0)
        masks = net.predict(images)
        #continue
        max_value = np.round(np.max(masks),2)
        min_value = np.round(np.min(masks),2)
        print("\nmax=",max_value,"min=",min_value)

        temp = masks[0]
        _,thr = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY)
        thr = thr.astype("uint8")
        temp = 1/(1+np.exp(-temp))#sigmoid

        #region 将mask转换为热力图
        temp = np.uint8(temp*255)
        img = np.uint8(img*255)
        temp = cv2.applyColorMap(temp,cv2.COLORMAP_JET)
        #endregion
        contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        h,w,c = img.shape
        dis = cv2.hconcat([img,temp])
        cv2.drawContours(dis, contours, -1, (0,0,255),3)
        cv2.putText(dis, data['filename'], (0,18), cv2.FONT_ITALIC, 0.7, (0,0,255), 1)
        cv2.putText(dis, f"max={str(max_value)},min={str(min_value)}", (0,h-2), cv2.FONT_ITALIC, 0.8, (0,0,255), 1)
        cv2.imshow("dis", dis)
        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default='./run/train_ic/best.pth', help='指定权重文件，未指定则使用官方权重！')  # 修改
    parser.add_argument('--out_path', default='./run/train_ic', type=str)  # 修改
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')

    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data_path', default='D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test')
    parser.add_argument('--data_path_val', default='')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--batch_size', default=16, type=int)

    opt = parser.parse_args()

    train(opt)
    #predict(opt)
