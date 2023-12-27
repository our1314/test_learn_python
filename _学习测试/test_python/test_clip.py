import torch
import numpy as np

a = np.clip(a=range(10), a_min=2, a_max=7)
print(a)


rr = torch.rand([423360, 128])
print(len(rr))