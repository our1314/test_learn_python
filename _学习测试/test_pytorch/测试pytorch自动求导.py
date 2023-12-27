""" 2022.10.9
"""
import torch
from torch import autograd


def test_标量对向量求导():
    x = torch.tensor(2.)  # 数字后的点表示省略0
    a = torch.tensor(2., requires_grad=True)  # 表示autograd.grad时，会对其进行求导，
    b = torch.tensor(2., requires_grad=True)
    c = torch.tensor(3., requires_grad=True)

    print("1、采用autograd.grad函数进行求导")
    y = a ** 2 * x + b * x + c  # **表示幂运算

    """ 梯度下降算法流程：
    1、给定初值
    2、计算初值的导数
    3、x1 = x0 + lr*d 得到新的初值，其中lr为步长，d为梯度。
    4、循环迭代2、3步,直到x1-x0很小，则得到y最小值对应的x
    如果将y视为loss,则利用梯度下降算法将得到使y最小时的卷积核、神经网络参数等等，参数规模为几十万级别。
    """

    # 梯度手动推导
    # dy/da = 2xa = 8   (其中x=2,a=2)
    # dy/db = x = 2     (其中x=2)
    # dy/dc = 1
    print('before:', a.grad, b.grad, c.grad)
    grads = autograd.grad(y, [a, b, c])
    print('after', grads[0], grads[1], grads[2])
    pass

    print("2、采用张量对象的方法backward进行求导")
    a.grad = None
    b.grad = None
    c.grad = None
    y = a ** 2 * x + b * x + c  # **表示幂运算
    y.backward()
    print('backward', a.grad, b.grad, c.grad)

    # 标量对向量求导，y = x1+x2+x3, dy/dx1 = 1
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = torch.sum(x)
    y.backward()
    print(f'标量对向量求导，手动推导结果为：[dy/dx1 = 1, dy/dx2 = 1, dy/dx3 = 1], pytorch计算结果为： {x.grad}')
    pass


# 复合函数求导
def test_复合函数求导():
    x = torch.tensor(1., requires_grad=True)  # 表示autograd.grad时，会对其进行求导，
    y = 2 * x + 1
    z = y ** 3

    # 复合函数求导,手动推导结果为：dz/dx = dz/dy * dy/dx = 3*y^2 * 2 = 3*(2*x+1)^2*2 = 54
    z.backward()
    print(f'复合函数求导,手动推导结果为：dz/dx = dz/dy * dy/dx = 3*y^2 * 2 = 3*(2*x+1)^2*2 = 54, pytorch计算结果为：x.grad = {x.grad}')
    print('y.grad =', y.grad)
    print('z.grad =', z.grad)
    pass


# 分段函数求导 http://www.pointborn.com/article/2021/7/21/1589.html
def test_分段函数求导():
    # x = torch.tensor([3., 4.], requires_grad=True)
    # if x > torch.tensor([1, 2]):
    #     y = torch.sum(x)
    # else:
    #     y = 0
    # y.backward()
    # print('x_grad', x.grad)
    pass


# 梯度下降算法求极小值，题目来至：《数值分析》裴玉茹译，2014，例13.6，计算结果一样，
def test_梯度下降求函数最小值():
    x = torch.tensor([1., -1.], requires_grad=True)
    lr = 0.1
    for i in range(200):
        f = 5 * x[0] ** 4 + 4 * x[0] ** 2 * x[1] - x[0] * x[1] ** 3 + 4 * x[1] ** 4 - x[0]
        # f.backward() 仅在叶节点中累积梯度。out不是叶节点，因此grad为None。autograd.grad可用于查找任何张量wrt到任何张量的梯度。
        grad = autograd.grad(f, x)
        grad = grad[0].data
        x1 = x - lr * grad
        if (x1 - x).norm()<1E-5:
            break
        x = x1
        #x.grad = None
        print(x)
        pass


if __name__ == '__main__':
    test_标量对向量求导()
    test_复合函数求导()
    test_分段函数求导()
    test_梯度下降求函数最小值()
