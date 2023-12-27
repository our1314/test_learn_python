#https://blog.csdn.net/hustlei/article/details/122634112

import numpy as np
import matplotlib.pyplot as plt

#0、创建容器
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

#1、生成离散坐标数据（x,y,z为二维数据）
x,y=np.mgrid[-2:2:0.2,-2:2:0.2]
z = x*np.exp(-x**2-y**2)

#2、绘制三维曲面
ax3d.plot_surface(x,y,z,cmap="YlOrRd")  #绘制3d螺旋线，参数x,y,z为一维数据
#ax3d.plot_wireframe 3D网格曲面
#ax3d.plot_trisurf 三角面
plt.show()
