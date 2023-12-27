# https://blog.csdn.net/weixin_39504171/article/details/106090626
import torch

a = torch.tensor([1, 2, 3])
c = a.expand(2, 3)
print(a)
print(c)


a = torch.tensor([1, 2, 3])
c = a.expand(3, 3)
print(a)
print(c)


a = torch.tensor([[1], [2], [3]])
print(a.size())
c = a.expand(3, 3)
print(a)
print(c)


a = torch.tensor([[1], [2], [3]])
print(a.size())
c = a.expand(3, 4)
print(a)


a = torch.tensor([[1], [2], [3]])
a = torch.unsqueeze(a, 0)
print(a.size())
c = a.expand(3, 3, 4)
print(a)
print(c)

# expand（）函数只能将size=1的维度扩展到更大的尺寸，如果扩展其他size（）的维度会报错。
# expand_as（）函数与expand（）函数类似，功能都是用来扩展张量中某维数据的尺寸，区别是它
# 括号内的输入参数是另一个张量，作用是将输入tensor的维度扩展为与指定tensor相同的size。


