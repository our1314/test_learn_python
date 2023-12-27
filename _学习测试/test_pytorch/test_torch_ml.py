import torch
import torch
from matplotlib import pyplot as plt

torch.manual_seed(100)
dtype = torch.float
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # unsqueeze 数据升维
y = 3 * x ** 2 + 2 + 0.2 * torch.rand(x.size())
plt.scatter(x.numpy(), y.numpy())
plt.show()

# 初始化权重参数
w = torch.randn(1, 1, dtype=dtype, requires_grad=True)
b = torch.zeros(1, 1, dtype=dtype, requires_grad=True)

# 训练模型
lr = 0.001
for ii in range(800):
    y_pred = x.pow(2).mm(w) + b
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()

    # 自动计算梯度，梯度存放在grad属性中
    loss.backward()

    # 手动更新参数，需要torch.no_grad()，使上下文环境中切断自动求导的计算
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # 梯度清零
        w.grad.zero_()
        b.grad.zero_()


# 可视化训练结果
plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-')
plt.scatter(x.numpy(), y.numpy(), c='blue', marker='o')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()
pass
