import torch
from torch import nn

nn.CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss();

temp_inputs = torch.rand((3, 2, 512, 512)).view(-1, 2)
temp_labels = torch.randint(1, (3, 512, 512)).view(-1)
CE_loss = loss_fn(temp_inputs, temp_labels)
pass
