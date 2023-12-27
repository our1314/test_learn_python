#!/usr/bin/python3
import argparse
import torch
import torchvision.transforms
from torchvision.utils import save_image
from gan.cyclegan.data import data_cyclegan, transform_A, transform_B
from gan.cyclegan.models import Generator


def test(opt):
    # Networks
    netG_A2B = Generator()
    netG_B2A = Generator()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load('output/netG_A2B.pth'))
    netG_B2A.load_state_dict(torch.load('output/netG_B2A.pth'))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    datasets_train = data_cyclegan(opt.data, transform_B, transform_B)

    for real_A, real_B in datasets_train:
        # Generate output
        fake_B = netG_A2B(real_A).data
        # fake_A = netG_B2A(real_B).data

        # Save image files
        torchvision.transforms.ToPILImage()(fake_B[0]).show()
        # torchvision.transforms.ToPILImage()(fake_A[0]).show()

        # save_image(fake_A, 'output/A.png')
        # save_image(fake_B, 'output/B.png')
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--out_path', default='./', type=str)  # 修改

    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data', default='D:/work/files/deeplearn_datasets/test_datasets/cycle_gan/train')  # 修改
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=2, type=int)

    parser.add_argument('--add_graph', default=False, type=bool)
    parser.add_argument('--save_period', default=20, type=int, help='多少轮保存一次，')
    parser.add_argument('--train_img', default=200, type=int, help='保存指定数量的训练图像')

    opt = parser.parse_args()

    test(opt)
