import torch
from torch import nn


class Loss_Fn(nn.Module):
    def __init__(self, class_num=2):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.],device=torch.device('cuda')), ignore_index=2)
        pass

    def forward(self, _out, _labels):
        data = _out.view(-1, 2)
        labels = _labels.view(-1)
        loss = self.loss_fn(data, labels)
        return loss


if __name__ == '__main__':
    loss_fn = Loss_Fn()
    temp_inputs = torch.rand((3, 2, 512, 512)).view(-1, 2)
    temp_labels = torch.randint(1, (3, 512, 512)).view(-1)

    loss = loss_fn(temp_inputs, temp_labels)
    print(loss)
