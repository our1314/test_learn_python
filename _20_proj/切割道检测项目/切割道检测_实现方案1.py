import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from our1314.work import GetAllFiles

def convimg(img):

    pass

"""对一维数据进行非极大值抑制"""
def 非极大值抑制(x):
    y = np.zeros_like(x)
    range_ = 2
    start = range_
    for i in range(start,x.shape[0]-start):
            r = x[i]  
            r1 = x[i-range_:i]
            r2 = x[i+1:i+range_+1]
            if r>np.max(r1) and r>=np.max(r2):
                y[i]=x[i]
    return y

"""将一维数据扩展后显示"""
def show_1dim(name, x, width=200):
    tmp = x.copy()
    tmp = np.expand_dims(tmp,1)
    tmp = np.repeat(tmp, width, axis=1)
    cv2.imshow(name,tmp)


if __name__ == "__main__":
    #D:\desktop\切割道测试图像
    files = GetAllFiles("D:/desktop/切割道测试图像")
    for f in files:
        print(f)
        src = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        srcH,srcW,_ = src.shape
        
        #1、原图二值化
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        t, bin = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV )#切割道为白色
        # kernel = np.ones((5,5),np.uint8)
        # bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel, iterations=1)#闭运算

        #2、连通域（筛选沟道，尺寸最大的为沟道，可使用图像分类）
        goudao = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin, connectivity=8, ltype=None)#第0个为背景
        for i in range(1,num_labels):#第0个为背景，因此从第一个开始
            sta, center = stats[i], centroids[i]  
            mask = np.zeros_like(labels,dtype=np.uint8)
            mask[labels==i] = 255
            #print(sta,center)
            if sta[2]==srcW and sta[3]==srcH:
                goudao = np.uint8(np.float32(labels==i)*255)#
                cv2.imshow("goudao", mask)
                #cv2.waitKey()
                break

        #3、对芯片区域进行形态学处理（闭运算、膨胀）
        goudao = cv2.bitwise_not(goudao)
        kernel = np.ones((5,5),np.uint8)
        goudao = cv2.morphologyEx(goudao, cv2.MORPH_CLOSE, kernel, iterations=1)#闭运算
        goudao = cv2.morphologyEx(goudao, cv2.MORPH_DILATE, kernel, iterations=1)#膨胀
        # # goudao = cv2.bitwise_not(goudao)
        cv2.imshow("ics", goudao)
        #cv2.waitKey()
        
        #4、对芯片区域提取连通域，筛选出芯片图片
        num_labels, labels = cv2.connectedComponents(goudao, connectivity=8, ltype=None)
        for i in range(1, num_labels):
            #A.生成与原图同尺寸的mask，并获取原图roi和mask roi
            mask = np.zeros_like(gray)
            mask[labels==i] = 255 
            x,y,w,h = cv2.boundingRect(mask)
            roi_src = src[y:y+h,x:x+w]
            roi_mask = mask[y:y+h,x:x+w] 
            # cv2.imshow("dis", roi_src)
            # cv2.waitKey()

            h,w = roi_mask.shape
            #print(max(h,w), min(h,w), max(h,w)/min(h,w))

            # if max(h,w)/min(h,w) >1.5:
            #     continue
            
            #B.图像转正
            pts = cv2.findNonZero(roi_mask)
            center,wh,angle = cv2.minAreaRect(pts)
            t = angle%90-90 if angle%90>45 else angle%90
            H = cv2.getRotationMatrix2D(center,t,scale=1)
            roi_src = cv2.warpAffine(roi_src, H, roi_src.shape[:2])
            roi_src = cv2.copyMakeBorder(roi_src, 20,20,0,0,cv2.BORDER_CONSTANT)

            #C.卷积
            img = roi_src
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (7,7), 1)

            k = np.arange(-1,1+1,2)
            kernel = k.reshape(len(k),1)*np.ones((1,250))
            kernel = kernel/np.abs(kernel).sum()
            dy = cv2.filter2D(img, cv2.CV_32F, kernel)
            dy = dy/np.max(np.abs(dy))

            #正方向导数
            dyp = np.zeros_like(dy)
            dyp[dy>0] = dy[dy>0]

            #反方向导数
            dyn = np.zeros_like(dy)
            dyn[dy<0] = np.abs(dy[dy<0])
            
            # cv2.imshow("dyp+dyn", cv2.hconcat([dyp,dyn]))

            #D.非极大值抑制
            dyp_ = dyp@np.ones((dyp.shape[1],1), np.float32)#投影
            dyp_ = dyp_.reshape(-1)/np.max(dyp_)#归一化
            yp = 非极大值抑制(dyp_)
            dis_yp = show_1dim("yp",yp)

            dyn_ = dyn@np.ones((dyn.shape[1],1), np.float32)#投影
            dyn_ = dyn_.reshape(-1)/np.max(dyn_)#归一化
            yn = 非极大值抑制(dyn_)
            dis_yn = show_1dim("yn",yn)

            # plt.plot(dyp_,"r")
            # plt.plot(y,"b")
            # plt.show()
            # print(dyp_)
            # print()


            # dyp = dyp/np.max(dyp)*255
            # dis = np.uint8(dyp)
            # dis = cv2.cvtColor(dis, cv2.COLOR_GRAY2BGR)

            # dis = np.uint8(dy/np.max(dy)*255)
            # h,w,_ = dis.shape
            # if len(max_edge)<2:
            #     continue


            y = yp+yn
            cnt=0
            for i in range(len(y)):
                h = i
                value = y[i]
                #print(value)
                if cnt == 0:
                    if value>0.3:
                        cv2.line(roi_src, (0,h), (w,h), (0,0,255))
                        cv2.putText(roi_src,str(round(value,2)),(0,h+cnt*30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
                        cnt += 1 
                else:
                    if value>1.5e-1:
                        cv2.line(roi_src, (0,h), (w,h), (0,0,255))
                        cv2.putText(roi_src,str(round(value,2)),(0,h+cnt*30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))
                        cnt += 1
                if cnt==2:
                    break
                
            #print(max_edge)
            # pos = max_edge[1]
            # dis = roi_src[0:pos,0:]
            
            cv2.imshow("src", roi_src)
            #cv2.imshow("dis", dis)
            #cv2.moveWindow("src",0,0)
            cv2.waitKey()
            pass
            
        
        cv2.destroyAllWindows()
