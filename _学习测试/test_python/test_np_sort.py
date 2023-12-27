import numpy as np
import torch

a = np.array([[[7,3,9,4],
                [2,5,3,7]],

               [[2,5,3,7],
                [7,3,9,4]]])
print(np.sort(a,0))

# a = torch.tensor([[
#                     [2,4,1],
#                     [7,5,9],
#                     [6,3,8]],
#                   [
#                     [1,4,1],
#                     [8,5,9],
#                     [6,3,8]]
#                     ])
# print('-'*100)
# print(torch.sort(a,dim=0))