import math
import torch
import torchvision
import torch.nn as nn
from resnet50 import backbone
class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone()
        self.rpn = None
        self.head = None

    def forward(self,x):
        x = self.backbone(x)
        