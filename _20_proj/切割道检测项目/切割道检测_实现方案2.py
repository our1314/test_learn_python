import cv2
from math import *
import numpy as np
from our1314.work import SE2, rad

"""
利用轻量级语义分割分割芯片区域
"""

if __name__ == "__main__":
    for i in range(10000):
        dis_width = 800
        grid_num = 10
        grid_d = dis_width/grid_num
        grid_w,grid_h = grid_d-20, grid_d-20
        grid_theta = i/10-50#np.random.rand()*60-30
        x = np.arange(-grid_num, grid_num+1)
        y = np.arange(-grid_num, grid_num+1)
        x,y = np.meshgrid(x,y)
        pts = np.stack([x.ravel(),y.ravel(),np.ones_like(x.ravel())])
        pts = SE2(dis_width/2,dis_width/2,rad(grid_theta)) @ np.diag([grid_d,grid_d,1]) @ pts
        pts = pts[0:2,:]

        dis = np.zeros((dis_width,dis_width,3),np.uint8) 
        pts_rect = np.array([[-grid_w/2,grid_w/2,grid_w/2,-grid_w/2],
                            [-grid_h/2,-grid_h/2,grid_h/2,grid_h/2],
                            [1,1,1,1]])
        
        for center in pts.T:
            H = SE2(center[0],center[1],rad(grid_theta))
            pts_rect_ = H@pts_rect
            pp = pts_rect_[0:2,:]
            pp = np.int32(pp.T)
            #cv2.polylines(dis, [pp], True, (0,0,255))
            cv2.fillPoly(dis, [pp], (255,255,255))
            
        cv2.imshow("dis", dis)
        cv2.waitKey(1)