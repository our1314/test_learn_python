import argparse
from datetime import datetime
import torch
import torchvision
from torch.utils.data import DataLoader
from data import data_seg, transform_val
from model import UNet


def detect(opt):
    # os.makedirs(opt.out_path, exist_ok=True)

    datasets_test = data_seg(opt.data_path, transform_val,
                             transform_val)

    dataloader_test = DataLoader(datasets_test, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    net = UNet()
    checkpoint = torch.load(opt.weights)
    net.load_state_dict(checkpoint['net'], strict=False)  # 加载checkpoint的网络权重
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"best.pth epoch: {epoch}, loss: {loss}")

    index = 0
    for img, label in dataloader_test:
        out = net(img)
        aa = out[0] + img[0]

        aa = (aa - torch.min(aa)) / (torch.max(aa) - torch.min(aa))
        img1 = torchvision.transforms.ToPILImage()(aa)
        img1.show()
        name = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")
        img1.save(f"{opt.out_path}/{name}.png")
        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/train/best.pth')
    parser.add_argument('--data_path',
                        default='D:/work/files/deeplearn_datasets/test_datasets/xray_real/val',
                        type=str)
    parser.add_argument('--out_path', default='./run/test', type=str)

    opt = parser.parse_args()
    detect(opt)
