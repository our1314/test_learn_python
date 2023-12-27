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

import torch
import torchvision
from torch.nn import Module
from torch import nn
import cv2
import sys
import torch.nn.functional as f
import os

sys.path.append("../")
# from _01_image_classification.cnn_imgcls.model import wide_resnet
from our1314.work import *
import torchvision.transforms as transforms

# class wide_resnet(Module):
#     def __init__(self, cls_num=2):
#         super(wide_resnet, self).__init__()
#         self.resnet =torchvision.models.wide_resnet50_2(pretrained=True)
#         self.resnet.fc = nn.Linear(2048, 2, bias=True)

#     def forward(self, x):
#         x = self.resnet(x)
#         # x = self.softmax(x)
#         return x

if __name__ == '__main__':
    #net = torchvision.models.resnet18(pretrained=True)
#     net = wide_resnet()
    net  = torchvision.models.wide_resnet50_2()

    path = "D:/work/program/python/DeepLearning/test_learn_python/_01_image_classification/cnn_imgcls/run/train/wide_resnet/weights/best.pth"
    if os.path.exists(path):
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['net'], strict=False)  # 加载checkpoint的网络权重
    #net = torch.load()
    net.eval()
    print(net)

    preprocess = transforms.Compose([   transforms.Resize((330,330)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    
    preprocess1 = transforms.Compose([  transforms.Resize((330,330)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    
    #1、设置前向传播和反向传播的回调函数，用于提取前向特征和梯度信息
    feature_map = []
    def forward_hook(module, fea_in, fea_out):
            feature_map.append(fea_out)
    net.layer1.register_forward_hook(hook=forward_hook)#提取layer4层的输出特征

    grad=[]
    def backward_hook(module, fea_in, fea_out):
         grad.append(fea_out)
    net.layer1.register_full_backward_hook(backward_hook)#提取梯度
    
    #2、打开图像
    orign_img = Image.open('D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/test1/ng/c (14).png').convert('RGB')
    orign_img = orign_img.resize((800,800))

    img = preprocess(orign_img)
    img2 = preprocess1(orign_img)
    orign_img = tensor2mat(img2)
    img = torch.unsqueeze(img, 0)

    #3、前向传播，获取最大分数，并反向传播提取梯度信息
    out = net(img)#前向传播
    cls = torch.argmax(out).item()
    score = out[:,cls].sum()
    net.zero_grad()
    score.backward(retain_graph=True)
    weights = grad[0][0].squeeze(0).mean(dim=(1,2))
    w = weights.view(*weights.shape, 1, 1)
    #w = torch.ones((512,1,1),dtype=float)

    #4、权重乘特征图
    fea = feature_map[0].squeeze(0)
    ss = w*fea #type:torch.Tensor
    ss = ss.sum(0)#在通道方向上进行求和（将通道维度压缩为1为，长宽尺寸不变，即保留了空间信息）
    
    #5、relu和归一化
    ss = f.relu(ss, inplace=True)#inplace表示覆盖原来的内存
    ss = ss/ss.max()   
    dd = ss.detach().numpy()
    
    #6、resize为图像尺寸，并转换为热力图
    aa = cv2.resize(dd, orign_img.shape[0:2])
    aa = np.uint8(aa*255)
    heatmap = cv2.applyColorMap(aa,cv2.COLORMAP_JET)
    heatmap = heatmap/255.0
    
    #7、将热力图与原图进行融合
    src = pil2mat(orign_img)
    dis = src*0.7 + heatmap*0.3
    
    print(cls)
    cv2.imshow('dis', dis)
    cv2.waitKey()
