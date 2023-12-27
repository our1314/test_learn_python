import cv2
import scipy
from scipy import interpolate
import torch
import numpy as np
import matplotlib.pyplot as plt
from our1314.work import *

"""
生成切割道训练图像思路：
1、生成矩形四个角点坐标
2、固定步长生成随机Y点坐标
3、样条插值连线
"""
def gen_rect1(len,off,num=20):
    #1
    pts1 = gen_line(len,off)
    pts1 = np.vstack([pts1, np.ones([1,pts1.shape[1]])])
    pts1  = SE2(-len//2,-len//2,0).dot(pts1)
    #4
    pts2 = gen_line(len,off)
    pts2 = np.vstack([pts2, np.ones([1,pts2.shape[1]])])
    pts2  = SE2(len//2,-len//2,pi/2).dot(pts2)
    pts = np.hstack([pts1,pts2])
    #2
    pts3 = gen_line(len,off)
    pts3 = np.vstack([pts3, np.ones([1,pts3.shape[1]])])
    pts3  = SE2(-len//2,len//2,0).dot(pts3)
    pts3 = np.fliplr(pts3)

    #3
    pts4 = gen_line(len,off)
    pts4 = np.vstack([pts4, np.ones([1,pts4.shape[1]])])
    pts4  = SE2(-len//2,-len//2,pi/2).dot(pts4)
    pts4 = np.fliplr(pts4)
    
    pts = np.hstack([pts1,pts2,pts3,pts4])
    return pts

def gen_line(len,off,num=20):
    _min,_max = off
    x = np.linspace(0,len,num)
    y = np.random.randint(_min,_max,size=num)
    f = interpolate.interp1d(x,y,kind='linear')
    
    x = np.arange(len)
    y = f(x)

    pts = np.stack([x,y],axis=0)
    return pts

for i in range(10**8):
    
    img = np.zeros([300,300,3], np.int32)
    mask1 = img.copy()
    mask2 = img.copy()
    h,w,c = img.shape
    theta = 0 #np.random.rand(1)*pi/60 - pi/60/2

    #1、绘制锯齿矩形
    pts = gen_rect1(len=200,off=(-3,3),num=20)
    pts = SE2(h/2,w/2,theta).dot(pts)
    pts = pts[0:2,:].T.astype(np.int32)

    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    if np.linalg.norm(np.array(list(color)))<20:
        continue
       
    cv2.fillPoly(img,[pts],color)
    cv2.fillPoly(mask1,[pts],(255,255,255))

    #2、绘制内部小矩形
    off = (-random.randint(0,1), 1)
    pts = gen_rect1(len=170,off=off,num=100)
    offx,offy = random.randint(-10,10),random.randint(-10,10)
    pts = SE2(h/2+offx, w/2+offy, theta).dot(pts)
    pts = pts[0:2,:].T.astype(np.int32)
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.fillPoly(img,[pts],color)#(0,0,0)
    cv2.fillPoly(mask2,[pts],(255,255,255))

    #3、绘制边线
    color = (random.randint(0,40),random.randint(0,40),random.randint(0,40))
    cv2.polylines(img, [pts], True, color, random.randint(1,2))
    
    noise = np.random.randn(h,w,c) * random.randint(5,10)
    img[img>0] = img[img>0] + noise[img>0]
    

    img[img>255]=255
    img[img<=0]=0
    img = img.astype("uint8")

    sigma = random.random()*1.2
    win = int(2*3*sigma+1)
    win = win if win%2==1 else win+1
    img = cv2.GaussianBlur(img, (win,win), sigmaX=sigma, sigmaY=sigma)
    

    #region 1、上边缘区域
    # mask = mask1 - mask2
    # mask = mask.astype("uint8")

    # roi_img = cv2.rectangle(img, [50-20,30,200+40,60], (255,255,255), 2)
    # roi_mask = cv2.rectangle(img, [50-20,30,200+40,60], (255,255,255), 2)
    # # roi_img = img[50-20,30,200+40,60]
    # # roi_mask = mask[50-20,30,200+40,60]
    # roi_img = img[30:90,30:270,:]
    # roi_mask = mask[30:90,30:270,:]
    # cv2.imshow("dis", roi_img)
    # cv2.waitKey(1)

    # os.makedirs("d:/desktop/qgd",exist_ok=True)
    # cv2.imwrite(f"d:/desktop/qgd/{i}.jpg", roi_img)
    # cv2.imwrite(f"d:/desktop/qgd/{i}.png", roi_mask)
    # continue
    #endregion

    #region 周边区域
    # os.makedirs("d:/desktop/qgd_around",exist_ok=True)
    # cv2.imwrite(f"d:/desktop/qgd_around/{i}.jpg", img)
    # cv2.imwrite(f"d:/desktop/qgd_around/{i}.png", mask)

    # cv2.imshow("dis",np.hstack([img, mask]))
    # cv2.waitKey(200)
    #endregion

    #region 中间区域
    os.makedirs("d:/desktop/qgd_center",exist_ok=True)

    noise = np.random.randn(h,w,c) * random.randint(15,25)
    img[mask2>0] = img[mask2>0] + noise[mask2>0]

    cv2.imwrite(f"d:/desktop/qgd_center/{i}.jpg", img)
    cv2.imwrite(f"d:/desktop/qgd_center/{i}.png", mask2)

    mask2 = mask2.astype("uint8")
    cv2.imshow("dis",np.hstack([img, mask2]))
    cv2.waitKey(1)
    #endregion