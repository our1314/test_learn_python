import torch
import torchvision
from torch import nn
from object_detection.手写yolov1.model.basic import SPP, CBL


class yolov1(nn.Module):
    def __init__(self, input_size=416, num_classes=2):
        super(yolov1, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        model = torchvision.models.resnet18(True)  # 官方代码生成resnet18
        self.backbone = nn.Sequential(*list(model.children())[:-2])  # 删除
        feat_dim = 512

        self.neck = nn.Sequential(
            SPP(),
            CBL(4 * feat_dim, feat_dim, kernel_size=1),
            CBL(feat_dim, feat_dim, kernel_size=3),
            CBL(feat_dim, feat_dim, kernel_size=3),
            CBL(feat_dim, feat_dim, kernel_size=3)
        )

        self.head = nn.Sequential(
            CBL(feat_dim, feat_dim // 2, kernel_size=1),  # //向下取整，即把结果的小数部分抹去
            CBL(feat_dim // 2, feat_dim, kernel_size=3, padding=1),
            CBL(feat_dim, feat_dim // 2, kernel_size=1),
            CBL(feat_dim // 2, feat_dim, kernel_size=3, padding=1),
        )

        # self.pred = nn.Conv2d(feat_dim, (4 + 1) * 2 + self.num_classes, kernel_size=1)
        self.pred = nn.Sequential(
            nn.Conv2d(feat_dim, (4 + 1) * 2 + self.num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)  # 主干网
        x = self.neck(x)  # 颈部网络
        x = self.head(x)  # 预测头
        x = self.pred(x)  # 预测层

        x = torch.permute(x, (0, 2, 3, 1))  # 交换维度[1,12,7,7] → [1,7,7,12]
        return x


if __name__ == '__main__':
    yolo = yolov1(416, 2)
    x = torch.rand(1, 3, 416, 416)
    y = yolo(x)

    print(y.shape)
    pass
