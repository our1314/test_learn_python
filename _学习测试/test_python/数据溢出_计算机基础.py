import numpy as np
import torch

a1 = np.array([200],np.uint8)
a2 = np.array([100],np.uint8)
a3 = a1 + a2
print(a3)
b1 = torch.tensor([200],dtype=torch.uint8)
b2 = torch.tensor([100],dtype=torch.uint8)
b3 = b1 + b2
print(b3)
