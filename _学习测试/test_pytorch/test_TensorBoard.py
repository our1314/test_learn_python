import torch.utils.tensorboard.writer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('log2')

for step in range(111):
    x = -50. + step
    writer.add_scalar("y=x^2 -2", x * x, step)
    pass
'''
进入log2的上一级目录，输入：tensorboard --logdir log2，得到查看结果的链接。
'''