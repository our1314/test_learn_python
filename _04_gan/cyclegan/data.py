import pathlib
import time
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision
import cv2
from torchvision.transforms import InterpolationMode

from image_segmentation.test_unet2.data import Resize1, Resize2, PadCenter

input_size = (256, 256)

transform_basic = [
    Resize1(),  # 边长变为偶数
    Resize2(240),  # 按比例缩放
    PadCenter(input_size),  # 四周补零
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    # torchvision.transforms.RandomVerticalFlip(0.5),
    # torchvision.transforms.RandomHorizontalFlip(0.5),

    # torchvision.transforms.RandomRotation(90, interpolation=InterpolationMode.NEAREST)
    # torchvision.transforms.RandomRotation(90, expand=False, interpolation=InterpolationMode.BILINEAR),
    # torchvision.transforms.CenterCrop(input_size),
]

transform_advan = [
    # torchvision.transforms.Pad(300, padding_mode='symmetric'),
    torchvision.transforms.GaussianBlur(kernel_size=(3, 7)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=(0.5, 1.5))
    # , contrast=(0.8, 1.2), saturation=(0.8, 1.2)),  # 亮度、对比度、饱和度
    # torchvision.transforms.ToTensor()
]
transform_A = torchvision.transforms.Compose(transform_basic + transform_advan)
transform_B = torchvision.transforms.Compose(transform_basic)


class data_cyclegan(Dataset):
    def __init__(self, data_path, transform_A=None, transform_B=None):
        self.transform_A = transform_A
        self.transform_B = transform_B

        A_path = os.path.join(data_path, "A")
        B_path = os.path.join(data_path, "B")

        ext = ['.jpg', '.png', '.bmp']

        self.A = [A_path + '/' + f for f in os.listdir(A_path) if
                  f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.B = [B_path + '/' + f for f in os.listdir(B_path) if
                  f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

    def __len__(self):
        return len(self.A)

    def __getitem__(self, item):
        # 取出图片路径
        A_path = self.A[item]
        B_path = self.B[item]

        image_A = cv2.imread(A_path, cv2.IMREAD_COLOR)  # type:cv2.Mat
        image_B = cv2.imread(B_path, cv2.IMREAD_COLOR)  # type:cv2.Mat

        image_A = cv2.cvtColor(image_A, code=cv2.COLOR_BGR2RGB)
        image_B = cv2.cvtColor(image_B, code=cv2.COLOR_BGR2RGB)

        if (self.transform_A is not None):
            image_A = self.transform_A(image_A)
        if (self.transform_B is not None):
            image_B = self.transform_B(image_B)
        return image_A, image_B


if __name__ == '__main__':
    d = data_cyclegan('D:/work/files/deeplearn_datasets/test_datasets/cycle_gan/train', transform_A, transform_B)

    data_loader = DataLoader(d, 1, True)
    for A, B in data_loader:
        torchvision.transforms.ToPILImage()(A[0]).show()
        time.sleep(0.1)
        torchvision.transforms.ToPILImage()(B[0]).show()
        time.sleep(0.1)

        # a = (b - torch.min(b)) / (torch.max(b) - torch.min(b))

        # seed = round(random.random() * 1000000000)
        # torchvision.transforms.ToPILImage()(img[0] + 0.3 * label[0]).show()
