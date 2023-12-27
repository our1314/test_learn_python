import os
import numpy as np
from math import *
from open3d import *
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries


def aa(a):
    return "Hello," + a


print("Start")
aa("xxx")

# points = np.random.rand(1000000, 3)  # 生成10000行，3列的随机数
# point_cloud = PointCloud()
# point_cloud.points = Vector3dVector(points)
# draw_geometries([point_cloud])

# print(point_cloud)

x0 = 0
y0 = 0
z0 = 0
r = 1

t = np.linspace(0, pi, 200)
p = np.linspace(0, 2 * pi, 200)
theta, phi = np.meshgrid(t, p)

x = x0 + r * np.sin(theta) * np.sin(phi)
y = y0 + r * np.sin(theta) * np.cos(phi)
z = z0 + r * np.cos(theta)

x = np.reshape(x, (40000, 1))
y = np.reshape(y, (40000, 1))
z = np.reshape(z, (40000, 1))

pts = np.hstack((x, y, z))
print(pts)

points = pts  # 生成10000行，3列的随机数
point_cloud = PointCloud()
point_cloud.points = Vector3dVector(points)
draw_geometries([point_cloud])
