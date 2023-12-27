import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.interpolate import interp1d
from scipy import interpolate
import cv2
import os
import random
from our1314.work import Utils

'''
https://www.blobmaker.app/
目标：生成与上述链接一样效果的随机图像
思路：
1、在极坐标系下随机生成theta和r
2、将坐标转换到笛卡尔坐标系下
3、采用滑动方式依次进行样条插值

https://www.codenong.com/33962717/
'''

def 生成随机二维形状():
    #1、在极坐标系下随机生成多个值
    num = 5
    theta = np.linspace(0, 2*pi - 2*pi/num, num)
    r = np.random.rand(num)*60+20
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    #2、转换到笛卡尔坐标系
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    #3、插值
    tck, u = interpolate.splprep([x, y], s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, 100), tck)
    
    #4、利用opencv绘制二维图像
    xi = xi - np.min(xi)
    yi = yi - np.min(yi)
    pts = np.stack([xi,yi],axis=0)
    w, h = int(np.max(xi))+1, int(np.max(yi))+1
    dis = np.zeros([h,w], dtype=np.uint8)
    pts = np.int32(pts).T
    dis = cv2.fillPoly(dis, [pts], 255)
    return dis
    #5、显示
    

#图像融合 
dir = 'D:/work/files/deeplearn_datasets/xray空洞检测/用于生成检测数据的图像2'
files_back = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith('.png')]
for i in range(0,500):

    #1、随机读取背景图，并扩展至目标尺寸
    idx = random.randint(0, len(files_back)-1)
    back = cv2.imdecode(np.fromfile(files_back[idx], dtype=np.uint8), cv2.IMREAD_ANYCOLOR)  # type:cv2.Mat
    scale = max(280/back.shape[0],280/back.shape[1])
    back = cv2.resize(back, dsize=None, fx=scale, fy=scale)
    h,w,_ = back.shape
    left,top = int((304-w)/2), int((304-h)/2)
    left = 0 if left <0 else left
    top = 0 if top <0 else top
    right,bottom = 304-w-left, 304-h-top
    right = 0 if right <0 else right
    bottom = 0 if bottom <0 else bottom
    back = cv2.copyMakeBorder(back, top, bottom, left, right, cv2.BORDER_CONSTANT)

    #2、随机生成指定数量的前景图，合并在一张mask图像上
    mask = np.zeros(back.shape[:2], np.uint8)
    for j in range(1):
        #a、生成指定数量前景图、resize
        force = 生成随机二维形状()
        tsize = random.randint(50,100)
        fh,fw = force.shape
        scale = min(tsize/fh,tsize/fw)
        force = cv2.resize(force, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
        #b、生成随机坐标(左上角)
        fh,fw = force.shape
        bh,bw = mask.shape
        x1,y1 = random.randint(0,bw-fw-1),random.randint(0,bh-fh-1)
        rect = [x1,y1,fw,fh]
        roi = mask[y1:y1+fh, x1:x1+fw]
        
        #c、合成mask图像
        mask[y1:y1+fh, x1:x1+fw] = force
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5),(1,1))#膨胀，填补空白
        # mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        # cv2.imshow("dis",mask)
        # cv2.waitKey()

    #3、融合
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #融合方式A（直接加权求和）
    #mix = np.uint8(back*0.9 + mask*(random.random()*0.1+0.1))

    #融合方式B（仅对mask部分加权求和）
    mix = back.copy()
    a = random.random()*0.2+0.08
    b = 1 - a
    mix[mask>0] = back[mask>0]*b+mask[mask>0]*a
    mix = np.uint8(mix)

    #4、显示
    dis = cv2.hconcat([back, mask, mix])
    cv2.putText(dis, os.path.basename(files_back[idx]), (0,100), cv2.FONT_ITALIC, 2, (0,0,255))
    cv2.imshow("dis", dis)
    cv2.waitKey(1)

    #5、保存
    from datetime import datetime
    name  = Utils.Now()
    path = 'D:/desktop/train'
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f'{path}/{name}.jpg', mix)
    cv2.imwrite(f'{path}/{name}.png', mask)