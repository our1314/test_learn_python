"""
操作步骤：
1、实例化网络模型
2、加载训练好的权重文件
3、输出某一层的特征图
4、截图特征图的通道转换为图像
"""

import torch
import torchvision
from torch.nn import Module
from torch import nn
import cv2
from torchvision import transforms
import sys
sys.path.append("../")
from myutils.myutils import *
from model import Model
    
if __name__ == '__main__':

    # preprocess = transforms.Compose([   transforms.Resize(256),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    preprocess = transforms.Compose([
                                        transforms.Resize((100,100)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
                                    
    fea = []
    def hookfn(module, fea_in, fea_out):
        fea.append(fea_out)

    #net = torchvision.models.resnet50(pretrained=True)
    net = Model()
    net.eval()
    print('*'*40)
    print(net)
    checkpoint = torch.load('best.pth')
    net.load_state_dict(checkpoint['net'])
    net.f[0].register_forward_hook(hook=hookfn)

    src = Image.open('D:/work/proj/抽检机/program/ChouJianJi/data/ic/2023-07-18_10.47.59-516.png').convert('RGB')
    x = preprocess(src)
    x = x.unsqueeze(0)    
    x = net(x)

    f = fea[0].squeeze(0)
    for i,ch in enumerate(f):
        mi = torch.min(ch)
        ma = torch.max(ch)
        ch = (ch-mi)/(ma-mi)#归一化至0-1
        ch = ch.unsqueeze(0)
        img = tensor2mat(ch)
        img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        img = cv2.applyColorMap(img,cv2.COLORMAP_JET)#转为热力图
        cv2.imshow(f'dis{i}', img)

        winH,winW = 1080,1920
        imgH,imgW = img.shape[0],img.shape[1]
        rowcnt = winW//imgW
        x = i%rowcnt
        y = i//rowcnt
        
        if y*imgH >= winH-imgH:
            break

        cv2.moveWindow(f'dis{i}',x*imgW,y*imgH)
        cv2.waitKey(1)

    cv2.waitKey()
    

    