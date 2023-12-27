
import numpy as np
import matplotlib.pyplot as plt

#1、生成离散坐标数据（x,y,z为二维数据）
x,y = np.mgrid[-10:10:0.1,-10:10:0.1]
z = np.cos(x+y) #np.cos(x) + np.cos(y)
#z = np.cos(x) + np.cos(y)

#2、绘制三维曲面
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
ax3d.plot_surface(x,y,z,cmap="YlOrRd")  #绘制3d螺旋线，参数x,y,z为一维数据
#ax3d.plot_wireframe(x,y,z)  #绘制3d螺旋线，参数x,y,z为一维数据
plt.show()
