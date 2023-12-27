import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def transform(img):
    img = transforms.ToTensor()(img)
    img = transforms.Resize(512, interpolation=InterpolationMode.NEAREST)(img)
    dis = torch.permute(img, (1, 2, 0))  # 交换维度
    cv2.imshow('dis', dis.numpy())
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    img = cv2.imdecode(np.fromfile('../test_CV/1.png', dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    transform(img)

    cv2.imshow('dis', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass
