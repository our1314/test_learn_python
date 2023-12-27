#https://blog.csdn.net/qq_43665602/article/details/126656713

import torch
from torch import nn

if __name__ == '__main__':
    inp = torch.tensor(
	   [[[[8, 6, 1, 6, 4],
          [1, 0, 0, 4, 7],
          [7, 2, 7, 6, 1],
          [5, 5, 3, 9, 4],
          [6, 2, 9, 5, 9]],

         [[2, 2, 6, 5, 3],
          [4, 3, 1, 7, 6],
          [4, 0, 5, 6, 3],
          [8, 2, 1, 6, 8],
          [6, 6, 7, 6, 1]],

         [[3, 4, 8, 7, 9],
          [7, 2, 4, 9, 3],
          [1, 2, 7, 9, 1],
          [1, 8, 5, 0, 0],
          [1, 0, 5, 1, 6]]]], dtype=torch.float32)
    
    print(inp.shape)# torch.Size([1, 3, 5, 5])
    for c in range(3):
        ch = inp[0, c, :, :]
        sum = torch.sum(ch)
        cnt = ch.shape[0] * ch.shape[1]
        print(f"mean channel {c}: {sum/(cnt)}")#计算每个通道的均值
    print('-'*25)

    out = nn.AvgPool2d(kernel_size=(inp.shape[2], inp.shape[3]))(inp)
    print(out)
    print(out.shape)