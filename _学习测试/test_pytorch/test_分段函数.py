"""
pytorch下可训练分段函数的写法：
https://blog.csdn.net/kuan__/article/details/118182946
"""
import torch

a = torch.tensor([-3, -1.5, 0.1, 2.3, 5], requires_grad=True)
b1 = a <= -2.5
b2 = (a <= -0.5) & (a > -2.5)
b3 = (a <= 0.5) & (a > -0.5)
b4 = (a <= 2.5) & (a > 0.5)
b5 = a > 2.5

pass

a1 = -1
a2 = 0.25 * a - 0.375
a3 = a
a4 = 0.25 * a + 0.375
a5 = 1

a1 = a1 * b1
a2 = a2 * b2
a3 = a3 * b3
a4 = a4 * b4
a5 = a5 * b5

c = a1 + a2 + a3 + a4 + a5
print(c)
pass
