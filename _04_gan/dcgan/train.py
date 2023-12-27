import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
from gan.dcgan.d_net import D_Net
from gan.dcgan.g_net import G_Net

# https://blog.csdn.net/bu_fo/article/details/109808354
if __name__ == '__main__':
    batch_size = 10
    if not os.path.exists("./dcgan_img"):
        os.mkdir("./dcgan_img")
    if not os.path.exists("./dcgan_params"):
        os.mkdir("./dcgan_params")
    input_size = (288, 288)
    s = input_size[0] / 184.0
    hh, ww = int(s * 106), int(s * 184)
    img_transf = transforms.Compose([
        transforms.Resize((hh, ww)),
        transforms.Pad(100, padding_mode='symmetric'),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    img_dir = r"D:/work/files/deeplearn_datasets/gan/sc70"
    # ImageFolder 不用自己写Dataset
    dataset = datasets.ImageFolder(img_dir, transform=img_transf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)

    d_weight_file = r"dcgan_params/d_net.pth"
    g_weight_file = r"dcgan_params/g_net.pth"
    if os.path.exists(d_weight_file) and os.path.getsize(d_weight_file) != 0:
        d_net.load_state_dict(torch.load(d_weight_file))
        print("加载判别器保存参数成功")
    else:
        d_net.apply(d_net.d_weight_init)
        print("加载判别器随机参数成功")

    if os.path.exists(g_weight_file) and os.path.getsize(g_weight_file) != 0:
        g_net.load_state_dict(torch.load(g_weight_file))
        print("加载生成器保存参数成功")
    else:
        g_net.apply(g_net.g_weight_init)
        print("加载生成器随机参数成功")

    loss_fn = nn.BCELoss()
    d_opt = torch.optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_opt = torch.optim.Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

    epoch = 1
    while True:
        print("epoch--{}".format(epoch))
        for i, (x, y) in enumerate(loader):
            # 判别器（判别器是一个分类器，1、对真实图像进行推理，期望其输出接近真实标签。2、对假图像进行推理，期望其输出接近假标签）
            real_img = x.to(device)
            # region
            # temp = real_img[0]
            # _, h, w = temp.shape
            # temp = temp * 0.5 + 0.5
            # a = int((h - hh) / 2)
            # img = temp[:, a:a + hh, :]
            # img1 = torchvision.transforms.ToPILImage()(img)
            # img1.show()
            # endregion

            real_label = torch.ones(x.size(0), 1, 1, 1).to(device)
            fake_label = torch.zeros(x.size(0), 1, 1, 1).to(device)

            real_out = d_net(real_img)  # 判别器对真实图像进行推理
            d_real_loss = loss_fn(real_out, real_label)  # 计算真实图像通过判别器的输出与真实标签值的损失（判别器是一个回归网络，希望判别器输出越来越接近真实标签）

            z = torch.randn(x.size(0), 128, 1, 1).to(device)
            fake_img = g_net(z).detach()  # 生成器生成假图像
            fake_out = d_net(fake_img)  # 判别器对假图像进行推理
            d_fake_loss = loss_fn(fake_out, fake_label)  # 计算假图像通过判别器的输出与假标签的损失

            d_loss = d_real_loss + d_fake_loss  # 判别器对真实图像的鉴别损失 + 假图像的鉴别损失
            d_opt.zero_grad()
            d_real_loss.backward()
            d_fake_loss.backward()
            d_opt.step()

            # 生成器（生成器，是一个生成图像的网络，其输出为一个假图像，将其给判别器进行推理，期望判别器输出为真实标签）
            fake_img = g_net(z)  # 生成器生成假图像

            # region
            # temp = fake_img[0]
            # temp = temp * 0.5 + 0.5
            # img1 = torchvision.transforms.ToPILImage()(temp)
            # img1.show()
            # endreion

            fake_out = d_net(fake_img)  # 判别器对假图像进行推理（鉴别）
            g_loss = loss_fn(fake_out, real_label)  # 计算假图像的输出与真实标签的损失
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i == 100:
                print("d_loss:{:.3f}\tg_loss:{:.3f}\td_real:{:.3f}\td_fake:{:.3f}".
                      format(d_loss.item(), g_loss.item(), real_out.data.mean(), fake_out.data.mean()))

                fake_image = fake_img.cpu().data
                save_image(fake_image, "./dcgan_img/{}_{}-fake_img.jpg".format(epoch, i), nrow=15, normalize=True,
                           scale_each=True)

        torch.save(d_net.state_dict(), "dcgan_params/d_net.pth")
        torch.save(g_net.state_dict(), "dcgan_params/g_net.pth")
        epoch += 1
