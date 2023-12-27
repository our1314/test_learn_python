#https://blog.csdn.net/hustlei/article/details/122634112

import numpy as np
import matplotlib.pyplot as plt

#0、创建容器
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

#1、生成坐标数据（x,y,z为一维数据）
theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
x = np.sin(theta)
y = np.cos(theta)
z = np.linspace(-2, 2, 100)

#2、绘图显示
ax3d.plot(x,y,z)  #绘制3d螺旋线，参数x,y,z为一维数据
plt.show()