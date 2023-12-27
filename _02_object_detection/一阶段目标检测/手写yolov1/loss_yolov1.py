import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def xywh2xyxy(xywh):
    xy, wh = xywh[:2], xywh[2:]
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    return x1y1


def compute_iou(box1, box2):
    """iou的作用是，当一个物体有多个框时，选一个相比ground truth最大的执行度的为物体的预测，然后将剩下的框降序排列，
    如果后面的框中有与这个框的iou大于一定的阈值时则将这个框舍去（这样就可以抑制一个物体有多个框的出现了），
    目标检测算法中都会用到这种思想。
    Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M]."""

    N = box1.size(0)
    M = box2.size(0)

    # torch.max(input, other, out=None) → Tensor
    # Each element of the tensor input is compared with the corresponding element
    # of the tensor other and an element-wise maximum is taken.
    # left top
    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    # right bottom
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)

    return iou


def compute_iou1(bbox1, bbox2):
    x1, y1, x2, y2 = xywh2xyxy(bbox1)
    a1, b1, a2, b2 = xywh2xyxy(bbox2)
    ax = max(x1, a1)  # 相交区域左上角横坐标
    ay = max(y1, b1)  # 相交区域左上角纵坐标
    bx = min(x2, a2)  # 相交区域右下角横坐标
    by = min(y2, b2)  # 相交区域右下角纵坐标

    area_bbox1 = (x2 - x1) * (y2 - y1)  # bbox1的面积
    area_bbox2 = (a2 - a1) * (b2 - b1)  # bbox2的面积

    w = max(0, bx - ax)
    h = max(0, by - ay)
    area_X = w * h  # 交集
    result = area_X / (area_bbox1 + area_bbox2 - area_X)
    return result


class loss_yolov1(nn.Module):
    def __init__(self, S=7, B=2, C=2, lambda_coord=5.0, lambda_noobj=0.1, device=torch.device("cuda")):
        super(loss_yolov1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.device = device

    def forward(self, pred_tensor, label_tensor):
        S, B, C = self.S, self.B, self.C
        N = B * 5 + C
        batch_size = pred_tensor.size(0)

        # 1、获取含有目标的索引
        obj_mask = label_tensor[:, :, :, 4] > 0  # 获取同样尺寸的mask,shape=[batch, 7, 7]
        obj_mask = obj_mask.unsqueeze(-1)  # 在末尾插入维度,shape=[batch, 7, 7, 1]
        obj_mask = obj_mask.expand_as(pred_tensor)  # 复制末端维度,shape=[batch, 7, 7, 12]

        # 2、获取不含有目标的索引
        noobj_mask = label_tensor[:, :, :, 4] == 0
        noobj_mask = noobj_mask.unsqueeze(-1)  # 在末尾插入维度
        noobj_mask = noobj_mask.expand_as(label_tensor)  # 复制末端维度

        # 3、从预测值中提取应该含有目标的数据，并分为：bbox数据和cls数据
        obj_pred = pred_tensor[obj_mask].view(-1, N)  # 从预测值中提取含有目标的数据，并reshape成若干行,12列
        bbox_pred = obj_pred[:, :5 * B].contiguous().view(-1, 5)  # 从含目标的数据中提取box信息，并reshape成(n,5)
        cls_pred = obj_pred[:, 5 * B:]  # 从含目标的数据中提取分类信息

        # 4、从label中提取有目标的数据,并分为：bbox数据和cls数据
        coobj_label = label_tensor[obj_mask].view(-1, N)
        bbox_label = coobj_label[:, :5 * B].contiguous().view(-1, 5)
        cls_label = coobj_label[:, 5 * B:]

        # 5、计算不含目标的损失
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)  # 从预测值中提取应该不含目标的数据
        noobj_label = label_tensor[noobj_mask].view(-1, N)  # 从标签中提取不含目标的数据
        noobj_pred_mask = torch.ByteTensor(noobj_pred.size())
        noobj_pred_mask.zero_()
        noobj_pred_mask[:, 4] = 1
        noobj_pred_mask[:, 9] = 1
        noobj_pred_conf = noobj_pred[noobj_pred_mask]  # 预测数据中不含目标的置信度
        noobj_label_conf = noobj_label[noobj_pred_mask]  # 标签中不含目标的置信度

        noobj_label_conf = noobj_label_conf.float()
        noobj_pred_conf = noobj_pred_conf.float()

        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_label_conf, reduce='sum')  # 计算无目标的损失

        # 6、计算含有目标的损失
        obj_response_mask = torch.ByteTensor(bbox_label.size()).to(self.device)
        obj_response_mask.zero_()
        obj_not_response_mask = torch.ByteTensor(bbox_label.size()).to(self.device)
        obj_not_response_mask.zero_()
        bbox_label_iou = torch.zeros(bbox_label.size()).to(self.device)
        for i in range(0, bbox_label.size()[0], 2):
            # 选择最佳IOU box # (x,y,w,h) → (x,y,x,y)
            box_pred_xywh = bbox_pred[i:i + self.B]  # 获取当前位置预测的两个box
            box_pred_xyxy = torch.FloatTensor(box_pred_xywh.size())
            box_pred_xyxy[:, :2] = box_pred_xywh[:, :2] / self.S - 0.5 * box_pred_xywh[:, 2:4]
            box_pred_xyxy[:, 2:4] = box_pred_xywh[:, :2] / self.S + 0.5 * box_pred_xywh[:, 2:4]

            box_label_xywh = bbox_label[i].view(-1, 5)
            box_label_xyxy = torch.FloatTensor(box_label_xywh.size())
            box_label_xyxy[:, :2] = box_label_xywh[:, :2] / self.S - 0.5 * box_label_xywh[:, 2:4]
            box_label_xyxy[:, 2:4] = box_label_xywh[:, :2] / self.S + 0.5 * box_label_xywh[:, 2:4]

            # 计算两个box与对应label的iou
            iou = compute_iou(box_pred_xyxy[:, :4], box_label_xyxy[:, :4])

            # label匹配到的box,在self.B个预测box中获取与label box iou值最大的那个box的索引
            max_iou, max_index = iou.max(0)
            obj_response_mask[i + max_index] = 1  # iou较大的索引
            obj_not_response_mask[i + 1 - max_index] = 1  # iou较小的索引

            bbox_label_iou[i + max_index, 4] = max_iou

        bbox_label_iou = Variable(bbox_label_iou)

        bbox_pred_reponse = bbox_pred[obj_response_mask].view(-1, 5).to(self.device)  # 提取较大IOU的bbox
        bbox_label_response = bbox_label[obj_response_mask].view(-1, 5).to(self.device)  # 提取对应label
        bbox_label_response_iou = bbox_label_iou[obj_response_mask].view(-1, 5).to(self.device)  # 提取对应iou

        # 含有目标的置信度损失
        loss_conf = F.mse_loss(bbox_pred_reponse[:, 4], bbox_label_response_iou[:, 4], reduction='sum')
        # bbox_pred_reponse[:, :2] = torch.sigmoid(bbox_pred_reponse[:, :2])
        loss_xy = F.mse_loss(bbox_pred_reponse[:, :2], bbox_label_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss((bbox_pred_reponse[:, 2:4]), (bbox_label_response[:, 2:4]), reduction='sum')
        loss_cls = F.mse_loss(cls_pred, cls_label, reduction='sum')

        # 总损失（无目标损失的权重较小，有目标损失权重较大）
        loss = self.lambda_noobj * loss_noobj + loss_conf + self.lambda_coord * (loss_xy + loss_wh) + loss_cls
        loss /= float(batch_size)
        return loss

    def compute_loss(pred_tensor, label_tensor):
        """
        计算无目标的损失、置信度损失、BBOX损失、分类损失
        1、无目标的网格，直接计算其置信度损失（label置信度为0）
        2、有目标的网格，需要计算其：
                                置信度损失
                                xy损失
                                wh损失
                                cls损失
        """
        pass


if __name__ == '__main__':
    pred = torch.rand((5, 7, 7, 12))
    label = torch.randint(0, 3, (5, 7, 7, 12))
    loss = loss_yolov1()

    a = loss(pred, label)
    pass
