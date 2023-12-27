import cv2
from math import *
import numpy as np

"""
2023.10.8
手写代码复现canny的非极大值抑制过程
整体思路是，遍历每个点，将该点与梯度方向上的相邻点进行比较，判断是否为极值点，是则保留，否则置零，遍历全图遍历一次即可得到效果。
（因为不能理解为什么只比较相邻点即可得到效果，为什么不比较梯度方向上更远的点，因此写代码测试。）

总结：将图像完整遍历一遍，记录下真正的极大值保存下来，即可实现非极大值抑制。
"""

if __name__ == "__main__":
    img = np.zeros((500,300),dtype=np.uint8)
    
    gray = [int(255*sin(pi/4*x)) for x in range(4+1)]
    for i in range(len(gray)):
        cv2.line(img, (0,100+i), (300,100+i), gray[i])

    for i in range(len(gray)):
        cv2.line(img, (0,104+i), (300,104+i), gray[i])

    for i in range(len(gray)):
        cv2.line(img, (0,300+i), (300,300+i), gray[i])
    
    cv2.imshow("dis", img)
    cv2.waitKey()

    maxedge = []
    for i in range(10,img.shape[0]-10):
        r2 = img[i:i+1, 0]
        near1 = img[i-5:i,0:1]
        near2 = img[i+1:i+5+1,0:1]
        near = np.vstack([near1,near2])
        if r2>np.max(near1) and r2>=np.max(near2):
            maxedge.append(i)
        # else:
        #     img[i+0:i+1,0:300]=0

    dis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(img.shape[0]):
        if i in maxedge:
            gray = img[i:i+1,0].item()
            cv2.line(dis, (0,i), (300,i), (gray,gray,gray))
        else:
            cv2.line(dis, (0,i), (300,i), (0,0,0))

    cv2.imshow("dis", dis)
    cv2.waitKey()