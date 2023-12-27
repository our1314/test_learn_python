from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#1、绘制简单二维函数
x = np.linspace(-np.pi, np.pi, 500)
y1,y2 = np.cos(x),np.sin(x)
plt.plot(x,y1)
plt.plot(x,y2)
plt.axis([0,6,0,6])#
plt.show()
 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
 
# Plot the surface
ax.plot_surface(x, y, z, color='b')
plt.show()

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure()
ax = fig.gca()
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()
 
plt.show()