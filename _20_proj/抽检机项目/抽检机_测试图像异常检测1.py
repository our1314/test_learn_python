import os
from random import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from model1 import Model1
from model2 import Model2
import torch
from torchvision import transforms

net = Model2()
net.eval()
#checkpoint = torch.load('best.pth')
#net.load_state_dict(checkpoint['net'])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100,100)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

path = 'D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/test/ng/0 (2).png'#input('输入图像路径：')
src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
dir_image = "D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/train/good"
files_all = os.listdir(dir_image)
images_path = [os.path.join(dir_image, f) for f in files_all if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
images_path = images_path[:100]

# images_path.append(path)
#shuffle(images_path)#随机排序
images_path.insert(0, path)

imgs = [cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR) for f in images_path]
#imgs = [cv2.medianBlur(cv2.resize(im, (100,100)), 3) for im in imgs]
imgs = np.array(imgs)
feas = imgs.copy()
feas = np.array([cv2.resize(im, None, fx=1, fy=1) for im in feas])

#region 神经网络提取特征
# IMGS = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
# IMGS = [test_transform(im) for im in IMGS]
# IMG = torch.stack(IMGS, dim=0)
# ff = net(IMG)
# ff = torch.permute(ff, (0,2,3,1))
# feas = ff.detach().numpy()
#endregion


#region RGB特征基础上添加额外特征
# gray = np.array([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in feas])
# gray = np.array([cv2.GaussianBlur(im, ksize=(7,7), sigmaX=1) for im in gray])
# sobelx = np.array([cv2.Sobel(im, cv2.CV_32F, 1, 0) for im in gray])
# sobely = np.array([cv2.Sobel(im, cv2.CV_32F, 0, 1) for im in gray])
# sobelx = sobelx[:,:,:,None]
# sobely = sobely[:,:,:,None]

# feas = np.concatenate((feas, sobelx, sobely), axis=3)
# feas = np.concatenate((feas, sobelx, sobely), axis=3)#只用梯度特征
#endregion

clf = LocalOutlierFactor(n_neighbors=40, contamination=0.01)#异常检测器


#region 遍历每个像素点，统计异常点所在的索引和坐标
result = np.zeros(feas.shape[:3])
_,rows,cols,_=feas.shape
for r in np.arange(rows):
    for c in np.arange(cols):
        a = feas[:,r,c,:]
        pred = clf.fit_predict(a)
        ng_index = [i for i,p in enumerate(pred) if p<0]
        for i in ng_index:
            result[i,r,c]+=1
            
for img, outlier in zip(imgs, result):
    outlier = cv2.normalize(outlier, 0, 255, norm_type=cv2.NORM_MINMAX)
    outlier = cv2.convertScaleAbs(outlier)
    outlier = cv2.applyColorMap(outlier, colormap=cv2.COLORMAP_JET)
    outlier = cv2.resize(outlier, img.shape[:2], interpolation=cv2.INTER_LINEAR)

    dis = np.hstack([img, outlier])
    cv2.imshow('dis', dis)
    cv2.waitKey()

cv2.destroyAllWindows()
#endregion


def onmouse(*p):
    event, x, y, flags, param=p
    if event == cv2.EVENT_LBUTTONDOWN:
        dis = src.copy()

        #1、绘制鼠标位置
        #cv2.circle(dis, (x,y), 5, (0,0,255), 2)
        cv2.line(dis,(x,y),(x,y),(0,0,255),1)
        cv2.imshow("dis", dis)

        #2、绘制曲线图
        b = imgs[:,y,x,0]
        g = imgs[:,y,x,1]
        r = imgs[:,y,x,2]
        #x = np.arange(0,imgs.shape[0])

        plt.clf()#清屏
        
        plt.subplot(311)
        plt.plot(b, c='blue')
        plt.subplot(312)
        plt.plot(g, c='green')
        plt.subplot(313)
        plt.plot(r, c='red')
        plt.legend([],[])
        plt.show()

        #3、异常检测
        # 1、收集异常点所在的图像索引
        # 2、所在图像索引上的异常坐标的聚类
        X = imgs[:,y,x,:]
        pred = clf.fit_predict(X)
        #pred = clf.score_samples(X)
        ng_index = [i for i,p in enumerate(pred) if p<0]
        print(ng_index)
        

cv2.namedWindow('dis')
cv2.setMouseCallback("dis",onmouse)
cv2.imshow('dis', src)
cv2.waitKey()

"""
总结：总体来说实现了思路，目前存在的问题如下
1、大量异常通常出现在图像未准确对齐的图像上
2、检测时间较长
解决方案：
1、正常测试时，图像想过应该比现在好很多，至少能保证不会出现过于模糊的图像。
2、图像对齐初步依靠模板匹配解决。
3、检测时间长的问题尝试使用深度学习训练的特征提取器提取特征进行解决。
第1、2在改造的机器上比较容易解决，现在优先测试第三个问题。
"""
