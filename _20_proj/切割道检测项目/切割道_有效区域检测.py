"""
一、割道边缘检测：
    思路1：
        a.过滤蓝色
        b.找到切割道边缘的梯度，指导canny算子的梯度
    思路2：
        a.分别沿XY方向计算梯度(幅值+方向+附近像素的梯度)，利用AI模型(如线性分类器)对其进行分类。
        b.将切割道附近的像素过滤出来分类，以避免计算量过大

二、芯片有效区域边缘检测：
    a.转换为梯度图像
    b.从切割道边缘开始滑窗+分类。
    c.非极大值抑制
"""

import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
from our1314.work.Utils import *
import sys
sys.path.append("D:/work/program/python/DeepLearning/test_learn_python/_10_proj/切割道检测项目")
from ransac_line import ransac_line

def 切割道图像转正(img):

    pass

# def ransac_line(pts):
#     choice = np.random.choice(len(pts),2)
#     p1,p2 = pts[choice]
#     pass


if __name__ == "__main__":
    path = "D:/work/files/deeplearn_datasets/其它数据集/切割道检测/芯片尺寸/6380B24L1805/10-111.jpg"
    src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    a,tmp = cv2.threshold(tmp, 20, 255, cv2.THRESH_BINARY)
    tmp = cv2.bitwise_not(tmp)

    # contours,hi = cv2.findContours(tmp,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # contours  = [np.squeeze(con,axis=1) for con in contours]
    # contours = np.vstack(contours)

    canny = cv2.Canny(tmp, 20, 60)
    #lines = cv2.HoughLines(canny, 1, pi/1800, 500)#, srn=2.0, stn=0)
    lines = cv2.HoughLinesP(canny, 1, 0.1*pi/180, 400, minLineLength=800, maxLineGap=2000)
    lines = np.squeeze(lines, axis=1)

    
    dis = src.copy()#cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    for line in lines:
        if len(line)==2:
            rho, theta = line
            x0,y0 = rho*cos(theta),rho*sin(theta)
            x1,y1 = int(x0-500000*cos(theta)),int(y0-sin(theta))
            x2,y2 = int(x0+500000*cos(theta)),int(y0+sin(theta))
            cv2.line(dis,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.imshow("dis",dis)
            cv2.waitKey()
        else:
            x1,y1,x2,y2 = line
            cv2.line(dis, (x1,y1), (x2,y2), (0,0,255), 1)

    cv2.imshow("dis",dis)
    cv2.waitKey()
    pass
    
    # ransac_line(contours,[1])

    # for con in contours:
    #     cv2.drawContours(img,con,-1,(0,0,255),1) 
    #     cv2.imshow("dis",img)
    #     cv2.waitKey()
    #     pass

