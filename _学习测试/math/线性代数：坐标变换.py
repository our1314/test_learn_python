import math
import numpy as np

vector_a = np.array([1, 2, 3, 4, 5])
vector_b = np.array([1, 2, 3, 4, 5])
result = vector_a * vector_b[:, None]
print('向量a与向量b的外积为：')
print(result)


pt = np.array([[5],[4],[1]])  # 二维点的齐次坐标
H = np.array(
    [
        [1,0,5],
        [0,1,4],
        [0,0,1]
    ])
print(H)
print(np.linalg.inv(H))

p = np.linalg.inv(H)@pt
print(p)