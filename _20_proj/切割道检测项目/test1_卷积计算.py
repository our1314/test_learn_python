import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as f

"""
旋转各个角度，找到水平卷积核响应最大的角度
"""

def conv2d(x, kernel):
    H,W = x.shape
    h,w = kernel.shape

    y = np.zeros((H-h+1,W-w+1),dtype=np.float32)

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            a1 = x[i:i+h,j:j+w].reshape(1,-1)
            a2 = kernel.reshape(-1,1)
            a = a1.dot(a2)
            y[i,j]=a
    return y

if __name__ == "__main__":
    img = cv2.imdecode(np.fromfile(file='D:/desktop/1.png',dtype=np.uint8), cv2.IMREAD_COLOR)
    #img = cv2.resize(img, (570,540))
    img = cv2.GaussianBlur(img, (7,7), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #k,img = cv2.threshold(img, 10,255, cv2.THRESH_TOZERO)

    img = np.float32(img)
    # img = F.to_tensor(img)#自动交换通道和归一化
    
    k = np.arange(-1,1+1,2)
    kernel = k.reshape(len(k),1)*np.ones((1,250))
    kernel = kernel/np.abs(kernel).sum()

    #卷积
    img = conv2d(img, kernel=kernel)#type:np.ndarray
    m1,m2 = np.max(img),np.min(img)

    img0 = img.copy()
    img0 = np.abs(img0)
    img0 = img0/np.max(img0)

    img1 = img.copy()
    img1[img1<0]=0
    img1 = img1/np.max(img1)

    img2 = img.copy()
    img2[img2>0]=0
    img2 = np.abs(img2)
    img2 = img2/np.max(img2)

    cv2.imshow("dis", cv2.hconcat([img1,img0,img2]))
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass