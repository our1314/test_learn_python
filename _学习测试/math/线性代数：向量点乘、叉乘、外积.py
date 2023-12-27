import numpy as np

#1、点乘和内积为统一概念，结果为一个标量
v1 = np.array([1,0,0])
v2 = np.array([0,1,0])
v3 = v1.dot(v2)
print(f'点乘：\n{v3}')

#2、叉乘，结果为一个向量，向量维度只能是2维或3维
v1 = np.array([1,0,0])
v2 = np.array([0,1,0])
v3 = np.cross(v1,v2)
print(f'叉乘：\n{v3}')

#3、外积，结果为一个矩阵，即Nx1矩阵乘1xM矩阵
v1 = np.array([1,2,3,4])
v2 = np.array([1,2,3,4,5])
v3 = np.outer(v1,v2)
print(f'外积：\n{v3}')

v1 = np.expand_dims(v1,axis=1)
v2 = np.expand_dims(v2,axis=0)
v3 = v1@v2
print(f'外积：\n{v3}')