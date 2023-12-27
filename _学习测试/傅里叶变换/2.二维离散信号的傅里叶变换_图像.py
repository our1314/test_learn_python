"""
绘制不同方向的正方形，计算幅频图

https://blog.csdn.net/qq_42856191/article/details/123776656
https://blog.csdn.net/dazhuan0429/article/details/85774692
https://zhuanlan.zhihu.com/p/110026009 K空间解释
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from our1314.work import *

#1、生成图像
def getimg(theta=45):
    #a.生成坐标
    x_ = np.linspace(-1,1,2,dtype=float)
    y_ = np.linspace(-1,1,2,dtype=float)
    x,y = np.meshgrid(x_,y_)
    x,y = x.ravel(),y.ravel()
    pts = np.stack([x,y,np.ones(x.shape[0])])
    p2 = pts[:,2].copy()

    pts[:,2] = pts[:,3]
    pts[:,3] = p2

    #b.坐标变换
    pts = SE2(400,400,rad(theta)) @ np.diag([200,200,1]) @ pts

    #c.绘图
    img = np.zeros((800,800,1), dtype=np.uint8)
    pts = np.int32(pts[0:2,:].T)
    cv2.fillPoly(img, np.array([pts]), (255,255,255))
    img = img/255.0
    return img

def 显示多角度傅里叶变换幅频图():
    for theta in range(45, 360*1000):
        #1、生成图像
        img = getimg(theta)

        #2、傅里叶变换
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dftShift = np.fft.fftshift(dft) #将图像中的低频部分移动到图像的中心
        fft_result = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))   # 将实部和虚部转换为实部，乘以20是为了使得结果更大
        print(dftShift[0,0],dftShift[400,400])

        img = np.float32(img) / 255.0
        result = fft_result/np.max(fft_result)

        cv2.imshow("dis", cv2.hconcat([img, result]))
        cv2.waitKey()

def 显示过滤频率后的傅里叶逆变换图像():
        #0、创建容器
        fig = plt.figure()
        ax2d = fig.add_subplot(1,3,1)  #创建2d坐标系
        ax2d_2 = fig.add_subplot(1,3,2)  #创建2d坐标系
        ax3d = fig.add_subplot(1,3,3, projection='3d')  #创建3d坐标系

        #1、生成图像
        img = getimg(15)
        
        #2、傅里叶变换
        fft_result = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        fft_result = np.fft.fftshift(fft_result) #将图像中的低频部分移动到图像的中心
        

        #region 显示fft结果
        fft_dis = 20 * np.log(cv2.magnitude(fft_result[:, :, 0], fft_result[:, :, 1]))   # 将实部和虚部转换为实部，乘以20是为了使得结果更大
        ax2d.imshow(fft_dis, cmap='gray')#cmap='gray' 不加默认为彩色图像
        plt.axis('off')
        #endregion


        R = 800//2
        for r in np.arange(1,R):
            fft_result_tmp = fft_result.copy()

            #3、按圆进行过滤
            circle_filt = np.zeros_like(fft_result_tmp)
            x0,y0 = circle_filt.shape[1]//2, circle_filt.shape[0]//2
            circle_filt = cv2.circle(circle_filt, (x0,y0), r, (1,1), -1)
            fft_result_tmp = circle_filt * fft_result_tmp
            fft_result_tmp[fft_result_tmp==0] = 1#

            #region 显示fft结果
            fft_dis = 20 * np.log(cv2.magnitude(fft_result_tmp[:, :, 0], fft_result_tmp[:, :, 1]))   # 将实部和虚部转换为实部，乘以20是为了使得结果更大
            ax2d_2.imshow(fft_dis, cmap='gray')#cmap='gray' 不加默认为彩色图像
            plt.axis('off')
            #endregion

            print('r=', r)

            #4、逆变换
            fft_result_tmp = np.fft.ifftshift(fft_result_tmp)  # 低频部分从图像中心移开
            iImg = cv2.idft(fft_result_tmp)              # 傅里叶反变换
            iImg = cv2.magnitude(iImg[:, :, 0], iImg[:, :, 1])  # 转化为空间域
            
            #5、显示图像
            s = 2**5
            x,y = np.mgrid[0:iImg.shape[1]:s, 0:iImg.shape[0]:s]
            iImg = cv2.resize(iImg, dsize=None, fx=1/s, fy=1/s)

            #plt.clf()
            ax3d = fig.add_subplot(1,3,3,projection='3d')  #创建3d坐标系
            ax3d.plot_surface(x, y, iImg, cmap="YlOrRd")
            plt.ion()
            plt.show()
            plt.axis('off')
            cv2.imshow("dis",img)
            cv2.waitKey()


if __name__ == "__main__":
    index = 1
    if index == 0:
        显示多角度傅里叶变换幅频图()
    else:
        显示过滤频率后的傅里叶逆变换图像()