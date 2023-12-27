"""
测试代码来至：《Python深度学习 基于PyTorch》2.6节
"""
# -*_ coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(100)  # 设置随机数种子
x = np.linspace(-1, 1, 100).reshape(100, 1)  # 生成100维x向量
y = 3 * np.power(x, 2) + 2 + 0.2 * np.random.rand(x.size).reshape(100, 1)  # 生成100维y向量，y=3x^2+2 添加高斯噪声

# 画图，显示x,y分布情况
plt.scatter(x, y)
plt.show()

# 随机初始化参数
w1 = np.random.rand(1, 1)
b1 = np.random.rand(1, 1)

'''
定义损失函数 Loss = 1/2*Sum(w*xi^2 + b - yi)
对损失函数求导：
dLoss/dw = 
dLoss/db = 

w1 = w1 - lr * dLoss/dw
b1 = b1 - lr * dLoss/db
'''
lr = 0.0001  # 学习率
for i in range(800):
    # 前向传播
    y_pred = x**2 * w1 + b1
    # 定义损失函数
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    # 定义梯度
    grad_w = np.sum((y_pred - y) * x**2)
    grad_b = np.sum((y_pred - y))
    # 使用梯度下降法，使loss最小
    w1 -= lr * grad_w
    b1 -= lr * grad_b
    print(w1, b1)

plt.ylabel('predict')
plt.plot(x, y_pred, 'r-')
plt.scatter(x, y, c='blue', marker='o')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()
print(w1, b1)
