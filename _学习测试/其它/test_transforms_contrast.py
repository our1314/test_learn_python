import cv2
import numpy as np
import torch
import torchvision
from myutils import myutils

img = torch.rand((3, 256, 256), dtype=torch.float32)  # type:torch.Tensor
minimum = img.amin(dim=(-2, -1), keepdim=True)

img = cv2.imdecode(np.fromfile('D:/desktop/XRAY-VISION2_2023-06-01_08.21.29-146.png', dtype=np.uint8), -1)  # type:cv2.Mat
img = myutils.mat2tensor(img)
RandomAutocontrast = torchvision.transforms.RandomAutocontrast(1)
RandomAutocontrast(img)