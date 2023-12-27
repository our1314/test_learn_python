#https://blog.csdn.net/hustlei/article/details/122634112

import numpy as np
import matplotlib.pyplot as plt

#1、绘制二维曲线
fig = plt.figure()
ax2d = fig.add_subplot()
theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
x = np.sin(theta)
y = np.cos(theta)
ax2d.plot(x,y)
plt.show()


ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
#2、绘制三维曲线
# theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
# x = np.sin(theta)
# y = np.cos(theta)
# z = np.linspace(-2, 2, 100)
# ax3d.plot(x,y,z)  #绘制3d螺旋线，参数x,y,z为一维数据
# plt.show()

#3、绘制三维曲面
x,y=np.mgrid[-2:2:0.2,-2:2:0.2]
z = x*np.exp(-x**2-y**2)
#ax3d.plot_wireframe(x,y,z)#绘制三维网格曲面，参数x,y,z为二维数据
ax3d.plot_surface(x,y,z,cmap="YlOrRd")
plt.show()
pass
