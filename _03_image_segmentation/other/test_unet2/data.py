import pathlib
import random
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision
from PIL import Image
import cv2
from torchvision.transforms import InterpolationMode
from global_val import seed

input_size = (512, 512)


# 将奇数尺寸图像更改为偶数尺寸
class Resize1():
    def __call__(self, img):
        h, w, c = img.shape

        W = w if w % 2 == 0 else w - 1
        H = h if h % 2 == 0 else h - 1

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return img


# 按比例缩放
class Resize2():
    def __init__(self, width):
        self.width = width

    def __call__(self, img):
        h, w, c = img.shape
        scale = self.width / max(w, h)
        W, H = round(scale * w), round(scale * h)
        dst = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        return dst


class PadCenter():
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w, c = img.shape
        W, H = self.size

        pad_top = round((H - h) / 2)
        pad_bottom = H - pad_top - h

        pad_left = round((W - w) / 2)
        pad_right = W - pad_left - w

        dst = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT)
        return dst


class MatToTensor():
    def __init__(self):
        self.totensor = torchvision.transforms.ToTensor()

    def __call__(self, img):
        dst = self.totensor(img)
        mmax = torch.max(dst)
        mmin = torch.min(dst)
        return dst


transform_basic = [
    Resize1(),  # 边长变为偶数
    Resize2(300),  # 按比例缩放
    PadCenter(input_size),  # 四周补零
    torchvision.transforms.ToTensor(),

    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomHorizontalFlip(0.5),

    torchvision.transforms.RandomRotation(90, interpolation=InterpolationMode.NEAREST)
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

trans_train_mask = torchvision.transforms.Compose(transform_basic)
trans_train_image = torchvision.transforms.Compose(transform_basic + transform_advan)

transform_val = torchvision.transforms.Compose([
    Resize1(),  # 边长变为偶数
    Resize2(300),  # 按比例缩放
    PadCenter(input_size),  # 四周补零
    torchvision.transforms.ToTensor()])


# transform_train = torchvision.transforms.Compose([
#
#     torchvision.transforms.Resize(input_size),
#     torchvision.transforms.Pad(100, padding_mode='symmetric'),
#     torchvision.transforms.RandomVerticalFlip(0.5),
#     torchvision.transforms.RandomHorizontalFlip(0.5),
#     # torchvision.transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
#     torchvision.transforms.RandomRotation(10, expand=False, interpolation=InterpolationMode.BILINEAR),
#     torchvision.transforms.RandomAffine(degrees=0, translate=(0.02, 0.01)),
#     torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 亮度、对比度、饱和度
#     # torchvision.transforms.ToTensor(),
#     torchvision.transforms.CenterCrop(input_size),
# ])


class data_seg(Dataset):
    def __init__(self, data_path, transform_image=None, transform_mask=None):
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        ext = ['.jpg', '.png', '.bmp']

        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "masks")

        self.Images = [images_path + '/' + f for f in os.listdir(images_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.Labels = [labels_path + '/' + f for f in os.listdir(labels_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放的同样尺度(256*256)。
        否则dataloader会报错。
        '''

        # 取出图片路径
        image_path = self.Images[item]
        label_path = self.Labels[item]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # type:cv2.Mat
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)  # type:cv2.Mat

        seed = []
        result_epoch_path = pathlib.Path(f'./run/train/seed.txt')
        with result_epoch_path.open('r') as fp:
            seed = int(fp.read())

        torch.manual_seed(seed)
        if self.transform_image is not None:
            image = self.transform_image(image)
        torch.manual_seed(seed)
        if self.transform_mask is not None:
            label = self.transform_mask(label)
        # else:
        #     h, w, c = image.shape
        #     scale = input_size[0] / max(h, w)
        #     th, tw = round(scale * h), round(scale * w)
        #
        #     image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        #     label = cv2.resize(label, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        #
        #     _, label = cv2.threshold(label, 0, 255, type=cv2.THRESH_BINARY)
        #
        #     pad_bottom = input_size[0] - th
        #     pad_right = input_size[1] - tw
        #
        #     image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT)
        #     label = cv2.copyMakeBorder(label, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT)
        #
        #     # cv2.imshow("dis1", image)
        #     # cv2.waitKey(1)
        #     #
        #     # cv2.imshow("dis2", label)
        #     # cv2.waitKey()
        #
        #     image = torchvision.transforms.ToTensor()(image)
        #     label = torchvision.transforms.ToTensor()(label)
        #
        #     # image = trans_train(image)
        #     # img = torchvision.transforms.ToPILImage()(image)  # type:PIL.Image.Image
        #     # img.show()

        return image, label


class SEGData(Dataset):
    def __init__(self, data_path):
        '''
        根据标注文件去取图片
        '''
        IMG_PATH = data_path + '/images'
        SEGLABE_PATH = data_path + '/masks'
        self.img_path = IMG_PATH
        self.label_path = SEGLABE_PATH
        # print(self.label_path)
        # print(self.img_path)
        # self.img_data = os.listdir(self.img_path)
        self.label_data = os.listdir(self.label_path)
        self.totensor = torchvision.transforms.ToTensor()
        self.resizer = torchvision.transforms.Resize((512, 512))

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放的同样尺度(256*256)。
        否则dataloader会报错。
        '''
        # 取出图片路径

        img_name = os.path.join(self.img_path, self.label_data[item].split('.p')[0])

        img_name = os.path.split(img_name)

        img_name = img_name[1] + '.png'
        # print(img_name+ "GGGGGGGGGGGG")
        img_data = os.path.join(self.img_path, img_name)
        label_data = os.path.join(self.label_path, self.label_data[item])
        # 将图片和标签都转为正方形
        # print("ggggggg" + label_data)
        img = Image.open(img_data)

        label = Image.open(label_data)
        w, h = img.size
        # 以最长边为基准，生成全0正方形矩阵
        slide = max(h, w)
        black_img = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_label = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_img.paste(img, (0, 0, int(w), int(h)))  # patse在图中央和在左上角是一样的
        black_label.paste(label, (0, 0, int(w), int(h)))

        # img.show()
        # label.show()
        # 变为tensor,转换为统一大小256*256
        img = self.resizer(black_img)
        label = self.resizer(black_label)

        # img.show()
        # label.show()
        # label1 = np.array(label)
        # img1 = np.array(img)
        # label = cv2.cvtColor(label1,cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        # label = label.reshape(1,label.shape[0],label.shape[1])
        # img = img.reshape(1,img.shape[0],img.shape[1])
        # i

        # cv2.imwrite("D:\\unet\\testData\\1.jpg",img1)
        # cv2.imwrite("D:\\unet\\testData\\2.jpg",label1)
        img = self.totensor(img)
        label = self.totensor(label)

        m1 = torch.max(img)
        m2 = torch.min(img)
        m3 = torch.max(label)
        m4 = torch.min(label)

        return img, label


if __name__ == '__main__':
    d = data_seg('D:/work/files/deeplearn_datasets/test_datasets/xray_real', transform_image=trans_train_image,
                 transform_mask=trans_train_mask)

    data_loader = DataLoader(d, 1, True)
    for img, label in data_loader:
        # torchvision.transforms.ToPILImage()(img[0]).show()
        # time.sleep(0.1)
        # torchvision.transforms.ToPILImage()(label[0]).show()
        # time.sleep(0.1)

        # a = (b - torch.min(b)) / (torch.max(b) - torch.min(b))

        #seed = round(random.random() * 1000000000)
        torchvision.transforms.ToPILImage()(img[0] + 0.3 * label[0]).show()
