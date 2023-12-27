import argparse
import os
from data import DatasetSplit,IMAGENET_MEAN,IMAGENET_STD
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import CJJDataset
from model import Teacher, Student, AutoEncoder, PDN_small
import datetime 
import random
import numpy as np
import cv2
import itertools
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchvision.transforms import functional as F
from our1314.work.Utils import addWeightedMask


def set_seeds(seed, with_torch=True, with_cuda=True):
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

@torch.no_grad()
def teacher_normalization(teacher, train_loader, device):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        train_image = train_image.to(device)

        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        train_image = train_image.to(device)

        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


def train(opt):
    os.makedirs(opt.out_path, exist_ok=True)
    set_seeds(42)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    full_train_set = CJJDataset(opt.data_path, split=DatasetSplit.TRAIN) #MVTecDataset(opt.data_path, "pill")
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(42)
    datasets_train, datasets_val = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)
    datasets_test = CJJDataset(opt.data_path, split=DatasetSplit.TEST)

    dataloader_train = DataLoader(datasets_train, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader_val = DataLoader(datasets_val, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader_test = DataLoader(datasets_test, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    teacher = Teacher(384)
    student = Student(384*2)
    autoencoder = AutoEncoder()

    teacher.to(device)
    student.to(device)
    autoencoder.to(device)

    #optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=opt.lr, weight_decay=1e-5)  # 定义优化器 momentum=0.99
    optimizer = torch.optim.SGD(itertools.chain(student.parameters(), autoencoder.parameters()), lr=opt.lr)  # 定义优化器 momentum=0.99

    # 学习率更新策略
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * opt.epoch), gamma=0.1)


    # 加载预训练模型
    loss_best = 9999
    if os.path.exists(opt.pretrain):
        checkpoint = torch.load(opt.pretrain)
        teacher.load_state_dict(checkpoint['net_teacher'])
        student.load_state_dict(checkpoint['net_student'])
        autoencoder.load_state_dict(checkpoint['net_autoencoder'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        time,epoch,loss = checkpoint['time'],checkpoint['epoch'],checkpoint['loss']
        #loss_best = checkpoint['loss']
        print(f"加载权重: {opt.pretrain}, {time}: epoch: {epoch}, loss: {loss}")
    
    p = torch.load("D:/desktop/EfficientAD-main/models/teacher_medium.pth")
    teacher.pdn.load_state_dict(p)
    teacher_mean, teacher_std = teacher_normalization(teacher, dataloader_train, device=device)

    for epoch in range(1, opt.epoch):
        # 训练
        teacher.eval()
        student.train()
        autoencoder.train()

        loss_train = 0
        for image_st, image_ae in dataloader_train:
            image_st = image_st.to(device)
            image_ae = image_ae.to(device)

            #student
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

            student_output_st = student(image_st)[:, :384]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            loss_st = loss_hard

            #autoencoder
            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

            student_output_ae = student(image_ae)[:, 384:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)

            loss_total = loss_st + loss_ae + loss_stae
            loss_train = loss_train + loss_total.item()

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
        scheduler.step()

        # 验证
        teacher.eval()
        student.eval()
        autoencoder.eval()

        # q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        #     validation_loader=dataloader_val, 
        #     teacher=teacher, 
        #     student=student,
        #     autoencoder=autoencoder, 
        #     teacher_mean=teacher_mean,
        #     teacher_std=teacher_std, 
        #     desc='Final map normalization',
        #     device=device)

        # auc = test(
        #     test_set=dataloader_test, 
        #     teacher=teacher, 
        #     student=student,
        #     autoencoder=autoencoder, 
        #     teacher_mean=teacher_mean,
        #     teacher_std=teacher_std, 
        #     q_st_start=q_st_start, 
        #     q_st_end=q_st_end,
        #     q_ae_start=q_ae_start, 
        #     q_ae_end=q_ae_end,
        #     test_output_dir=opt,
        #     desc='Final inference',
        #     device=device)
        
        # 打印一轮的训练结果
        #loss_train = loss_train / len(dataloader_train)
        #loss_val = loss_val / len(dataloader_val)
        
        loss_train = loss_train/len(dataloader_train.dataset)
        print(f"epoch:{epoch}, loss_train:{round(loss_train, 6)}, auc:{round(0, 6)}, lr:{optimizer.param_groups[0]['lr']}")


        # 保存权重
        if loss_train < loss_best:
            loss_best = loss_train

            teacher.eval()
            student.eval()
            autoencoder.eval()

            checkpoint = {'net_teacher': teacher.state_dict(),
                          'net_student': student.state_dict(),
                          'net_autoencoder': autoencoder.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'teacher_mean':teacher_mean,
                          'teacher_std':teacher_std,
                          'epoch': epoch,
                          'loss': loss_train,
                          'time': datetime.date.today()}
            torch.save(checkpoint, os.path.join(opt.out_path,opt.weights))
            print(f'已保存:{opt.weights}')
    
def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None, desc='Running inference', device=None):
    y_true = []
    y_score = []
    for image, label in tqdm(test_set, desc=desc):
        orig_width = 256
        orig_height = 256

        image = image.to(device)

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        # if test_output_dir is not None:
        #     img_nm = os.path.split(path)[1].split('.')[0]
        #     if not os.path.exists(os.path.join(test_output_dir, defect_class)):
        #         os.makedirs(os.path.join(test_output_dir, defect_class))
        #     file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
        #     tifffile.imwrite(file, map_combined)

        y_true_image = 0 if label[0] == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)#图像经过teacher网络得到输出
    teacher_output = (teacher_output - teacher_mean) / teacher_std#对teacher网络的输出进行归一化
    student_output = student(image)#图像经过student网络得到输出
    autoencoder_output = autoencoder(image)#图像经过autoencoder网络得到输出

    map_st = torch.mean((teacher_output - student_output[:, :384])**2, dim=1, keepdim=True)#求teacher网络输出与student网络输出的差，并求均值
    map_ae = torch.mean((autoencoder_output - student_output[:, 384:])**2, dim=1, keepdim=True)#求autoencoder网络输出与student网络输出的差，并求均值
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae   

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, desc='Map normalization', device=None):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        image = image.to(device)
        map_combined, map_st, map_ae = predict(image=image, teacher=teacher, student=student, autoencoder=autoencoder, teacher_mean=teacher_mean, teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

def mytest(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datasets_test = CJJDataset(opt.data_path, split=DatasetSplit.TEST)
    dataloader_test = DataLoader(datasets_test, batch_size=opt.batch_size, shuffle=False, drop_last=True)

    teacher = Teacher()
    student = Student(384*2)
    autoencoder = AutoEncoder()

    checkpoint = torch.load(opt.pretrain)
    teacher.load_state_dict(checkpoint['net_teacher'])
    student.load_state_dict(checkpoint['net_student'])
    autoencoder.load_state_dict(checkpoint['net_autoencoder'])
    teacher_mean,teacher_std = checkpoint['teacher_mean'],checkpoint['teacher_std']
    time,epoch,loss = checkpoint['time'],checkpoint['epoch'],checkpoint['loss']
    print(f"加载权重: {opt.pretrain}, {time}: epoch: {epoch}, loss: {loss}")

    teacher.eval()
    student.eval()
    autoencoder.eval()

    teacher.to(device)
    student.to(device)
    autoencoder.to(device)
    teacher_mean = teacher_mean.to(device)
    teacher_std = teacher_std.to(device)

    for image,label in dataloader_test:
        image = image.to(device)
        map_combined, map_st, map_ae = predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None)
        
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (256, 256), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        # F.to_pil_image(map_combined*255).show()
        mask = map_combined
        txt = f"max={str(np.round(np.max(mask),3))},min={str(np.round(np.min(mask),3))}"
        
        img = image.to('cpu')#type:torch.Tensor
        img = img.squeeze(dim=0)
        img = img.numpy()
        img = img.transpose([1,2,0])
        img = img*IMAGENET_STD + IMAGENET_MEAN
        img = img.astype("float32")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # thr = mask.astype('uint8')
        # _,thr = cv2.threshold(thr, 0.5, 1, cv2.THRESH_BINARY)
        # contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (0,0,255),3)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        dis = cv2.hconcat([img, mask])
        dis = cv2.putText(dis, txt, (0,250), cv2.FONT_ITALIC, 1, (0,0,255))
        cv2.imshow('dis',dis)
        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default='./run/train/best.pth', help='指定权重文件，未指定则使用官方权重！')  # 修改
    parser.add_argument('--out_path', default='./run/train', type=str)  # 修改
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')

    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data_path', default='D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test')
    parser.add_argument('--data_path_val', default='')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)

    opt = parser.parse_args()

    #train(opt)
    mytest(opt)
    #predict(opt)
