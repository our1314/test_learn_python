import torch
from torch.nn import Linear, Module
from torchvision.models.resnet import resnet18, ResNet18_Weights, wide_resnet50_2
from our1314.work import exportsd


class net_xray(Module):
    def __init__(self, pretrained, cls_num=2):
        super(net_xray, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = Linear(512, cls_num, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        # x = self.softmax(x)
        return x

class wide_resnet(Module):
    def __init__(self, cls_num=2):
        super(wide_resnet, self).__init__()
        
        self.resnet = wide_resnet50_2(pretrained=True)
        self.resnet.fc = Linear(2048, 2, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        # x = self.softmax(x)
        return x

if __name__ == '__main__':
    input_shape = (1, 3) + data_wide_resnet.input_size
    img = torch.randn(size=input_shape)
    print(img.shape)
    net = net_xray(True)
    print(net)
    out = net(img)
    # print(f'out shape: {out.shape}')

    save_path = 'D:/desktop/net_xray.dat'
    f = open(save_path, "wb")
    exportsd.save_state_dict(net.to("cpu").state_dict(), f)
    f.close()
