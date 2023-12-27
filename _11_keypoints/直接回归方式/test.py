import argparse
import os
import pathlib
import cv2
from PIL import Image
import numpy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from net import net_resnet18, net_resnet50
from data_keypoints import data_keypoints
import torchvision

def test(opt):
    # 定义设备
    device = torch.device("cpu")
    net = net_resnet18()
    path = './run_resnet18/weights/best.pth'
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    print("loss:",checkpoint["loss"])
    net.to(device)
    net.eval()

    path = 'D:/work/files/deeplearn_datasets/choujianji/roi-keypoint/images'#"D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/test2/ok"#
    files = [path+"/"+ f for f in os.listdir(path)]

    for image_path in files:
        # 验证
        with torch.no_grad():
            img = Image.open(image_path).convert("RGB")
            img = torchvision.transforms.ToTensor()(img)
            img = torchvision.transforms.Resize((300,300))(img)

            out = net(img.unsqueeze(dim=0))

            shape = img.shape
            ww = shape[2]
            hh = shape[1]

            out = torch.flatten(out)
            x1 = (out[0] * ww).item()
            y1 = (out[1] * hh).item()
            x2 = (out[2] * ww).item()
            y2 = (out[3] * hh).item()
            x3 = (out[4] * ww).item()
            y3 = (out[5] * hh).item()
            x4 = (out[6] * ww).item()
            y4 = (out[7] * hh).item()

            pt1 = (int(x1),int(y1))
            pt2 = (int(x2),int(y2))
            pt3 = (int(x3),int(y3))
            pt4 = (int(x4),int(y4))
            
            img = img.numpy() * 255  # type:numpy.ndarray
            img = img.swapaxes(0, 2)
            img = img.swapaxes(0, 1)
            img = img.astype(np.uint8)

            # image = np.zeros((512, 512, 3), dtype=np.uint8)
            # print(type(img), type(image))
            # cv2.rectangle(image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_8, 3)  # 红
            img = img.copy()
            # cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(img, pt1, 3, (0,0,255), -1)
            cv2.circle(img, pt2, 3, (0,0,255), -1)
            cv2.circle(img, pt3, 3, (0,0,255), -1)
            cv2.circle(img, pt4, 3, (0,0,255), -1)

            cv2.imshow("dis", img)
            cv2.moveWindow("dis",0,0)
            cv2.waitKey()
            # img = torchvision.transforms.functional.to_pil_image(imgs[0])
            # Image.show(img)

    

        pathlib.Path(f'{opt.model_save_path}/weights').mkdir(parents=True,
                                                             exist_ok=True)  # https://zhuanlan.zhihu.com/p/317254621
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--model_save_path', default='run/train', help='save to project/name')

    opt = parser.parse_args()
    test(opt)
