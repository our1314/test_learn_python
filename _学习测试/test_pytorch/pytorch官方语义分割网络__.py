import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, DeepLabHead

if __name__ == '__main__':
    net = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights)
    net.classifier = DeepLabHead(2048, 1)
    