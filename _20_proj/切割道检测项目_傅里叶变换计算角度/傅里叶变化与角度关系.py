"""
绘制不同方向的正方形，计算幅频图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from our1314.work import *

for i in range(45,360*1000):
    #1、生成坐标
    x_ = np.linspace(-1,1,2,dtype=float)
    y_ = np.linspace(-1,1,2,dtype=float)
    x,y = np.meshgrid(x_,y_)
    x,y = x.ravel(),y.ravel()
    pts = np.stack([x,y,np.ones(x.shape[0])])
    p2 = pts[:,2].copy()
    
    pts[:,2] = pts[:,3]
    pts[:,3] = p2
    # plt.plot(pts[0,:],pts[1,:],".r")
    # plt.show()

    #2、坐标变换
    pts = SE2(400,400,rad(i)) @ np.diag([200,200,1]) @ pts

    #3、绘图
    dis = np.zeros((800,800,1), dtype=np.uint8)
    pts = np.int32(pts[0:2,:].T)
    cv2.fillPoly(dis, np.array([pts]), (255,255,255))
    #dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)

    #4、傅里叶变换
    dft = cv2.dft(np.float32(dis), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft) #将图像中的低频部分移动到图像的中心
    fft_result = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))   # 将实部和虚部转换为实部，乘以20是为了使得结果更大
    print(dftShift[0,0],dftShift[400,400])

    dis = np.float32(dis) / 255.0
    result = fft_result/np.max(fft_result)

    cv2.imshow("dis", cv2.hconcat([dis, result]))
    cv2.waitKey()
    # cv2.imshow("dis",dis)
    # cv2.waitKey()