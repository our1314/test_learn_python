import torch
import torchvision.models.resnet
from torch.nn import Module, Linear, Sigmoid
from torchvision.models.resnet import resnet101, resnet50, resnet34, resnet18


class net_resnet18(Module):
    def __init__(self):
        super(net_resnet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = Linear(512, 4, bias=True)  # 更改全连接层
        self.sigmoid = Sigmoid()  # 没有Sigmoid时网络输出为nan，网络难以收敛！
        print(self.resnet)  # 打印网络模型

        total_params = sum(p.numel() for p in self.resnet.parameters())
        print(f'总参数数量：{total_params}')
        total_trainable_params = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad == True)
        print(f'可训练参数数量：{total_trainable_params}')

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)  # sigmoid将目标数据归一化到0~1，使得网络很容易收敛。
        return x


if __name__ == '__main__':
    img = torch.randn(size=(3, 3, 2048, 2448))
    net = net_resnet18()
    out = net(img)

    print(f'img shape: {img.shape}')
    print(f'shape: {out.shape}')
    pass
