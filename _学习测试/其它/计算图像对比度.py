#https://blog.csdn.net/zsc201825/article/details/89645190

import cv2
import numpy as np


def contrast(img0):
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    m,n = img1.shape

    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE)/1.0
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1, cols_ext-1):
            b += (img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 + (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2
            cg = b / (4*(m-2)*(n-2) + 3*(2*(m-2) + 2*(n-2)) + 2*4)
    print(cg)

def contrast(img):
    k=np.array[
        [0,-1,0],
        [-1,1,-1],
        [0,-1,0]]
    a = cv2.filter2D(img, -1, k)

img0 = cv2.imread('d:/desktop/qxj/2023-11-16_02.44.50-685.png')
contrast(img0)