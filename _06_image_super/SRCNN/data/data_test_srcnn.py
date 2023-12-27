"""
2022.9.26
从文件夹读取数据
提供dataloader
数据增强
将数据加载到GPU

dataset 为数据集
dataloader 为数据集加载器
"""
import os
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torch.nn.functional as f

input_size = (214, 115)
#input_size = (256, 697)
class_num = 2

# trans_train = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(input_size),
#     torchvision.transforms.Pad(100, padding_mode='symmetric'),
#     # torchvision.transforms.GaussianBlur(kernel_size=(3, 15), sigma=(0.1, 15.0)),  # 随机高斯模糊
#     torchvision.transforms.ColorJitter(brightness=(0.3, 3), contrast=(0.5, 1.5), saturation=0.9),  # 亮度、对比度、饱和度
#     torchvision.transforms.RandomRotation(10, expand=False, interpolation=InterpolationMode.BILINEAR),
#     torchvision.transforms.CenterCrop(input_size),
#     torchvision.transforms.ToTensor(),
# ])
#
# trans_val = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(input_size),
#     torchvision.transforms.ToTensor(),
# ])


class data_srcnn(Dataset):
    def __init__(self, data_path, num_channels=3):
        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "labels")

        self.Images = [images_path + '/' + f for f in os.listdir(images_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.Labels = [labels_path + '/' + f for f in os.listdir(labels_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        image = Image.open(image_path)
        label = Image.open(label_path)

        image = image.convert("RGB")
        label = label.convert("RGB")

        image = torchvision.transforms.ToTensor()(image)
        label = torchvision.transforms.ToTensor()(label)
        return image, label


datasets_train = data_srcnn("D:/work/files/deeplearn_datasets/test_datasets/xray_super")
datasets_val = data_srcnn("D:/work/files/deeplearn_datasets/test_datasets/xray_super")

if __name__ == '__main__':
    dataloader_train = DataLoader(datasets_train, 1, shuffle=True)
    dataloader_val = DataLoader(datasets_val, 1, shuffle=True)

    max_width = 0
    max_height = 0
    for imgs, labels in dataloader_val:
        img1 = imgs[0, :, :, :]
        img1 = torchvision.transforms.ToPILImage()(img1)
        img1.show()

        img2 = labels[0, :, :, :]
        img2 = torchvision.transforms.ToPILImage()(img2)
        img2.show()

    print(max_width, max_height)
