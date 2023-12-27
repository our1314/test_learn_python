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
    return img


if __name__ == "__main__":
    img = getimg()
    
    #opencv 傅里叶变换
    fft_opencv = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fftShift_opencv = np.fft.fftshift(fft_opencv) #将图像中的低频部分移动到图像的中心
    fft_opencv = 20 * np.log(cv2.magnitude(fftShift_opencv[:, :, 0], fftShift_opencv[:, :, 1])) #将实部和虚部转换为实部，乘以20是为了使得结果更大

    #numpy 傅里叶变换
    fft_numpy = np.fft.fft2(np.float32(img))
    fftshift_numpy = np.fft.fftshift(fft_numpy)
    fft_numpy = 20*np.log(np.abs(fftshift_numpy))

    plt.subplot(121), plt.imshow(fft_opencv, cmap='gray')
    plt.title('opencv')
    plt.subplot(122), plt.imshow(fft_numpy, cmap='gray')
    plt.title('numpy')
    plt.axis('off')
    plt.show()

