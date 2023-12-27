import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)
vgg16.classifier.add_module('7', nn.Linear(1000, 10))

intput_data = torch.ones(1, 3, 48, 48)
output = vgg16.classifier(intput_data)

print(vgg16)
print(vgg16.features)
print(vgg16.avgpool)
print(vgg16.classifier)

vgg16.classifier = nn.Linear(25088, 10)
print(vgg16)
pass
