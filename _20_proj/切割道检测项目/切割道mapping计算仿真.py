'''
问题建模：
在一网格上画一个圆，输出圆附近、圆内的网格中点坐标
1、圆的参数已知（拟合）
2、还需要知道的参数有：
    a.网格点参数（网格点长/宽）
    b.圆心在网格点的偏移量

提前标定：
采集三个坐标点
1、计算相机参数
2、计算位姿关系
需要采集的参数点有：
1、采集圆弧坐标点
2、采集直线坐标点

3、移动至圆心位置，选定并记录原点O坐标
4、往X方向移动，记录X1点坐标，并记录芯片数量
5、往Y方向移动，记录Y1点坐标，并记录芯片数量

6、计算出圆心后，移动至圆心位置，根据图像计算网格中心与圆心的偏移量（需要标定相机）
7、计算当前芯片中心格子与圆心的偏移量

'''
from our1314.work import Utils
import numpy as np
import cv2
from math import *

def DrawBuffer():
    #芯片尺寸
    icw,ich = 200,300

    #晶圆直径
    r = np.int32(3100/2)

    img = np.zeros((4000,4000,3), np.uint8)
    imgh,imgw,_ = img.shape

    x = np.arange(0,4000,icw)
    y = np.arange(0,4000,ich)

    for xi in x:
        cv2.line(img,(xi,0),(xi,4000),(255,255,255),3)
    for yi in y:
        cv2.line(img,(0,yi),(4000,yi),(255,255,255),3)

    center = (2100,2030)
    cv2.ellipse(img,center,(r,r),0,-150,150,(0,0,255),3)
    cv2.circle(img,center,10,(0,0,255),-1)
    p1 = (np.int32(center[0]+r*np.cos(-150*pi/180)),np.int32(center[1]+r*np.sin(-150*pi/180)))
    p2 = (np.int32(center[0]+r*np.cos(150*pi/180)),np.int32(center[1]+r*np.sin(150*pi/180)))
    cv2.line(img,p1,p2,(0,0,255),3)

    cv2.imshow("dis", cv2.resize(img, (1000,1000)))
    cv2.waitKey()

def CalCoord():
    px,py = 0.00345,0.00345
    theta = 3*pi/180
    H = np.array([
        [cos(theta),-sin(theta),1000],
        [sin(theta),cos(theta),2000],
        [0,0,1]])
    
    

DrawBuffer()