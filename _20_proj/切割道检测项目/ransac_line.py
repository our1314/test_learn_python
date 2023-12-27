import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
from our1314.work.Utils import *
from time import time

def ransac_line(pts,choice):
    '''
    随机选择两个点，计算其余点离直线的距离，如果小于某个阈值，则认为是内点，
    自适应ransac
    初始N为无穷
    随机选两个点，计算内点率，再计算外点绿，再计算一个N
    参考《计算机视觉》鲁鹏
    '''
    N = 10**1000#迭代次数
    e = 0.5#外点率
    p = 0.999#找到目标直线的概率
    s = 2

    debug_cnt = 0#记录迭代次数

    start = time()
    thr = 0.5
    inner_cnt = 0
    best_line = []
    for i in range(N):
        pt1,pt2 = pts[np.random.choice(len(pts),2)]
        d = np.array([点到直线的距离(pt1, pt2, pt) for pt in pts])

        cnt = len(d[d<thr])#计算内点数量
        if cnt > inner_cnt:
            inner_cnt = cnt
            best_line = np.array([pt1,pt2])
            
            #自适应迭代次数处理
            e = 1 - inner_cnt/len(pts)#计算当前直线的外点率
            N = log(1-p)/log(1-(1-e)**s)#计算迭代次数
            i = 0
        if i > N*5:
            break
        debug_cnt += 1
    end = time()
    print("耗时：",end-start)

    line = 两点式转一般式(*best_line)
    best_line = 绘制直线(line)
    plt.plot(pts[:,0],pts[:,1],'.g')
    plt.plot(pts[choice][:,0],pts[choice][:,1],'.b')
    plt.plot(pts[60:,0],pts[60:,1],'.b')
    plt.plot(best_line[:,0],best_line[:,1],'r-')
    plt.show()
    pass

def gen_line_pts():
    cnt_sum = 60
    cnt_noise = 30

    #1、生成直线，并随机抽取坐标点
    t = np.linspace(0,100,cnt_sum)
    #t = np.random.choice(t,50)
    theta = pi/3
    x0,y0 = 0,0
    x = x0 + t*cos(theta)
    y = y0 + t*sin(theta)
    x = np.expand_dims(x,axis=0)
    y = np.expand_dims(y,axis=0)
    pts = np.vstack([x,y])
    pts = pts.T

    #2、沿着直线垂直方向添加高斯噪声
    vec_line = np.array([[cos(theta),sin(theta)]])
    vec_line = SO2(pi/2).dot(vec_line.T)#垂直于直线的方向向量

    noise = (np.random.randn(cnt_noise)*1*vec_line).T
    choice = np.random.choice(cnt_sum,cnt_noise)
    pts[choice] = pts[choice] + noise

    #3、随机添加外点噪声
    outlier = np.random.randint(0,100,[10,2])
    pts = np.vstack([pts,outlier])

    return pts,choice

def 两点式转一般式(pt1,pt2):
    x1,y1 = pt1
    x2,y2 = pt2

    A = y2-y1
    B = x1-x2
    C = x2*y1-x1*y2

    return np.array([A,B,C])

def 点到直线的距离(pt1,pt2, pt):
    line = 两点式转一般式(pt1,pt2)
    A,B,C = line
    pt = np.append(pt,1)
    d = abs(line.dot(pt))/np.sqrt(A*A+B*B)
    return d

def 绘制直线(line):
    A,B,C = line
    y1 = -A/B*0-C/B
    y2 = -A/B*60-C/B
    pt1,pt2 = [0,y1],[60,y2]
    return np.array([pt1,pt2])


if __name__ == "__main__":
    for i in range(100):
        pts,choice = gen_line_pts()
        ransac_line(pts,choice)
        pass