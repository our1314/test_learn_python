from torch import nn


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, input, target):#input=out, target=labels
        """
            计算多个batch的dicc
            @param input: 模型预测值，Shape:[B, C, W, H]
            @param target: one_hot形式的标签，Shape:[B, C, W, H]
        """
        batch_num, class_num = input.shape[0: 2]

        assert input.size() == target.size(), "the size of predict and target must be equal."

        # 计算交集
        intersection = (input * target).reshape((batch_num, class_num, -1)).sum(dim=2)

        # 计算并集
        union = (input + target).reshape((batch_num, class_num, -1)).sum(dim=2)

        dice = (2. * intersection + 1) / (union + 1)

        dice = dice.mean()

        loss = 1 - dice

        return loss
