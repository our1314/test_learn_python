'''
Grad-CAM (Gradient-weighted Class Activation Mapping) 是一种可视化深度神经网络中哪些部分对于预测结果贡献最大的技术。
https://baijiahao.baidu.com/s?id=1763572174465139226&wfr=spider&for=pc

https://www.jianshu.com/p/fd2f09dc3cc9
论文《Learning Deep Features for Discriminative Localization》发现了CNN分类模型的一个有趣的现象：
CNN的最后一层卷积输出的特征图，对其通道进行加权叠加后，其激活值（ReLU激活后的非零值）所在的区域，即为图像中
的物体所在区域。而将这一叠加后的单通道特征图覆盖到输入图像上，即可高亮图像中物体所在位置区域。

作者：西北小生_
链接：https://www.jianshu.com/p/fd2f09dc3cc9
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''

import numpy as np
import torch
import torchvision
from torch.nn import Module
from torch import nn
import cv2
import sys
import torch.nn.functional as f
import os

sys.path.append("../")
from our1314.work import *
import torchvision.transforms as transforms

if __name__ == '__main__':
    net = torchvision.models.resnet18(pretrained=True)
    net.eval()

    preprocess = transforms.Compose([   transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    preprocess1 = transforms.Compose([  transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    
    
    feature_map = []
    def forward_hook(module, fea_in, fea_out):
            feature_map.append(fea_out)
    
    #0、提取layer4层的输出特征(原图像经过layer4输出的特征图)
    net.layer4.register_forward_hook(hook=forward_hook)
    
    #1、读取图像并预处理
    orign_img = Image.open('image/dog2.jpg').convert('RGB')
    img = preprocess(orign_img)

    img2 = preprocess1(orign_img)
    orign_img = tensor2mat(img2)

    img = torch.unsqueeze(img, 0)#插入维度

    #2、前向传播
    with torch.no_grad():
        out = net(img)

    #3、获取分类类别
    cls = torch.argmax(out).item()
    score = torch.max(f.softmax(out, dim=1)).item()

    #4、提取全连接层的在此类别上的权重(行向量)
    weights = net.fc.weight.data[cls,:]
    tt=type(weights.shape)
    print(weights.shape)
    print(*weights.shape)
    w = weights.view(*weights.shape, 1, 1)

    #5、将权重与特征相乘
    fea = feature_map[0].squeeze(0)
    ss = w*fea #type:torch.Tensor
    ss = ss.sum(0)
    
    #6、将小于0的值置0，并缩放至0-1
    ss = f.relu(ss, inplace=True)#inplace表示覆盖原来的内存
    ss = ss/ss.max()

    #7、将特征图缩放至图像大小
    aa = cv2.resize(ss.numpy(), orign_img.shape[0:2])
    aa = np.uint8(aa*255)
    heatmap = cv2.applyColorMap(aa, cv2.COLORMAP_JET)
    heatmap = heatmap/255.0
    
    src = pil2mat(orign_img)
    dis = src*0.5+heatmap*0.5
    #dis = cv2.convertScaleAbs(dis)
    #dis = cv2.addWeighted(src,0.5,heatmap,0.5,gamma=0) #orign_img*0.5+heatmap*0.5

    print(cls)
    cv2.imshow('dis', dis)
    cv2.waitKey()
    pass