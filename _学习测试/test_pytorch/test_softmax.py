import torch
import torch.nn.functional as F

'''
softmax公式为:
'''
input = torch.tensor([0.6, -2.3, 1.9])
out = F.softmax(input)
print(f'input:{input}')
print(f'out:{out}')