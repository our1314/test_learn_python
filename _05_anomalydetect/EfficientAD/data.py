import torch
import os
import PIL
from enum import Enum
from torchvision.transforms import transforms
import cv2
from pathlib import Path


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
        imagesize=256,
        split=DatasetSplit.TRAIN,
    ):
        super().__init__()
        self.source = source
        self.split = split

        self.imgpaths = self.get_image_data()

        self.default_transform = transforms.Compose([
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.transform_ae = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2)
        ])

    def __getitem__(self, idx):
        image_path = self.imgpaths[idx]
        image = PIL.Image.open(image_path).convert("RGB")

        if self.split == DatasetSplit.TRAIN or self.split == DatasetSplit.VAL:
            return self.default_transform(image), self.default_transform(self.transform_ae(image))
        else:
            label = os.path.basename(os.path.dirname(image_path))
            return self.default_transform(image),label

    def __len__(self):
        return len(self.imgpaths)

    def get_image_data(self):
        data_dir = os.path.join(self.source, self.split.value)
        #imgpaths = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        
        imgpaths = []
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


if __name__ ==  "__main__":
    data = CJJDataset('D:/work/files/deeplearn_datasets/anomalydetection/mvtec_anomaly_detection/bottle',split=DatasetSplit.TEST)
    for d in data:
        img_st,a = d
        img_st = img_st.numpy()
        img_st = img_st.transpose([1,2,0])
        img_st = img_st*IMAGENET_STD + IMAGENET_MEAN
        img_st = img_st.astype("float32")
        img_st = cv2.cvtColor(img_st, cv2.COLOR_RGB2BGR)
        cv2.imshow("dis",img_st)
        cv2.waitKey()
        torch.utils.data.random_split()