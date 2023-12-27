import torch
import os
import PIL
from enum import Enum
from torchvision.transforms import transforms
import cv2

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CJJDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        resize=300,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0.05,
        contrast_factor=0.1,
        saturation_factor=0.1,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.train_val_split = train_val_split

        self.imgpaths = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees, 
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            
            transforms.GaussianBlur(kernel_size=(7,7),sigma=(0.1,2.0)),#随机高斯模糊          

            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        if split == DatasetSplit.TRAIN:
            self.transform = transforms.Compose(self.transform_img)
        else:
            self.transform = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)


    def __getitem__(self, idx):
        image_path = self.imgpaths[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {
                "image": image,
                "filename": os.path.basename(os.path.dirname(image_path)),
                }

    def __len__(self):
        return len(self.imgpaths)

    def get_image_data(self):
        data_dir = os.path.join(self.source, self.split.value)
        imgpaths = []#[os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        self.get_filelist(data_dir, imgpaths)
        return imgpaths
    
    def get_filelist(self, dir, Filelist):
        newDir = dir
        if os.path.isfile(dir):
            Filelist.append(dir)
            # # 若只是要返回文件文，使用这个
            # Filelist.append(os.path.basename(dir))
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                # 如果需要忽略某些文件夹，使用以下代码
                #if s == "xxx":
                    #continue
                newDir=os.path.join(dir,s)
                self.get_filelist(newDir, Filelist)

        return Filelist


if __name__ == "__main__":
    #data = MVTecDataset('d:/work/files/deeplearn_datasets/anomalydetection/test1', "pill",split=DatasetSplit.TEST)
    data = CJJDataset('D:/work/files/deeplearn_datasets/anomalydetection/mvtec_anomaly_detection/aaa',split=DatasetSplit.TEST)
    for d in data:
        img = d['image']
        img = img.numpy()
        img = img.transpose([1,2,0])
        img = img*IMAGENET_STD + IMAGENET_MEAN
        img = img.astype("float32")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("dis",img)
        cv2.waitKey()
        