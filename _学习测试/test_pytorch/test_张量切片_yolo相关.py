import torch

pi = torch.rand((1, 3, 80, 80, 6))
b = torch.randint(1, (24, 1)).view(-1)
a = torch.randint(1, (24, 1)).view(-1)
gi = torch.randint(10, (24, 1)).view(-1)
gj = torch.randint(10, (24, 1)).view(-1)
c = pi[b, a, gj, gi]
pass
