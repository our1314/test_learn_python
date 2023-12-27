import os.path
import random
import time
import PIL
import torch.optim
import torchvision.datasets
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.model = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((8,8)),#导出到onnx有问题
        #     nn.AvgPool2d((64,75)),
        #     nn.Flatten(),
        #     nn.Linear(3*8*8,10),
        #     nn.ReLU(),
        #     nn.Linear(10, 3),
        #     nn.Softmax()
        # )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,stride=1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,stride=1),
            nn.Flatten(),
            nn.Linear(147808, 400),
            nn.ReLU(),
            nn.Linear(400, 3),
            # nn.Softmax(),
        )


    def forward(self, x):
        x = self.model(x)
        return x

test_data = torch.randn((1,3,512,612))
print(test_data.shape)
net = Net()
dd = net(test_data)
print(dd.shape)

# region 0、初始化配置参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = './out'
if not os.path.exists(model_path):
    os.mkdir(model_path)

is_train = False
writer = SummaryWriter("logs")
# endregion

# region 1、初始化数据
datasets = torchvision.datasets.ImageFolder('D:\\桌面\\有无识别数据集2022.6.20', transform=torchvision.transforms.Compose([
    # torchvision.transforms.Resize(128),

    torchvision.transforms.RandomRotation(180),
    # torchvision.transforms.RandomGrayscale(0.1),
    torchvision.transforms.ToTensor()
]))
n = len(datasets)
n_test = random.sample(range(1, n), int(0.2 * n))  #按比例取随机数列表 https://blog.csdn.net/TycoonL/article/details/125667592
dataset_mnist_test = torch.utils.data.Subset(datasets, n_test)
dataset_mnist_train = torch.utils.data.Subset(datasets, list(set(range(1, n)).difference(set(n_test))))
# dataset_mnist_train = torch.myutils.data.Subset(data, range(int(0.8 * len(data))))
# dataset_mnist_test = torch.myutils.data.Subset(data, range(int(0.2 * len(data))))

print(len(dataset_mnist_train))
print(len(dataset_mnist_test))

dataloader_train = DataLoader(dataset_mnist_train, 8, shuffle=True)
dataloader_test = DataLoader(dataset_mnist_test, 5, shuffle=True)

print(len(dataloader_train))
print(len(dataloader_test))
# endregion

if is_train:
    # region
    model = Net()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=0.01)
    # opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # opt = torch.optim.Adam(model.parameters(), lr=0.01)
    # endregion

    model.train()
    # region 选择加载已训练过的模型还是新建模型
    '''
    断点继续参考：
    https://www.zhihu.com/question/482169025/answer/2081124014
    '''
    current_epoch = 0
    lists = os.listdir(model_path)
    if not len(lists) == 0:
        lists.sort(key=lambda fn: os.path.getmtime(model_path + "\\" + fn))  # 按时间排序
        last_pt_path = os.path.join(model_path, lists[len(lists) - 1])
        checkpoint = torch.load(last_pt_path)
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        model.train()
    # endregion

    # ******************************************************************************
    for epoch in range(current_epoch, 150):
        train_loss = 0
        train_acc = 0
        eval_loss = 0
        eval_acc = 0

        model.train()
        for images, labels in dataloader_train:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_acc += ((outputs.argmax(1) == labels).sum()/images.shape[0]).item()

        train_loss = train_loss / len(dataloader_train)
        train_acc = train_acc / len(dataloader_train)

        writer.add_scalar('loss', train_loss, epoch)
        writer.add_scalar('acc', train_acc, epoch)

        print(f"epoch：{epoch}, train_loss：{train_loss}, train_acc：{train_acc}")

        if epoch % 5 == 0:
            state_dict = {'net': model.state_dict(),
                          'optimizer': opt.state_dict(),
                          'epoch': epoch}
            p = f'{model_path}/{time.strftime("%Y.%m.%d_%H.%M.%S")}-epoch={epoch}-loss={str(round(train_loss, 5))}-acc={str(round(train_acc, 5))}.pth'
            torch.save(state_dict, p)

    # 保存整个模型（包括网络参数和网络结构）
    # torch.save(model.state_dict(), model_path)

else:
    lists = os.listdir(model_path)
    if len(lists) != 0:
        pass

    lists.sort(key=lambda fn: os.path.getmtime(model_path + "\\" + fn))
    last_pt_path = os.path.join(model_path, lists[len(lists) - 1])
    checkpoint = torch.load(last_pt_path)

    model = Net()
    model.load_state_dict(checkpoint['net'])
    model.eval()
    out = torch.tensor([0])

    acc = 1.
    for images, labels in dataloader_test:
        out = model(images)
        _, index = out.max(1)

        # print((index == labels).sum())
        num_correct = (index == labels).sum().item()
        # print(num_correct)
        acc = num_correct / images.shape[0]
        print(acc)
        pass

        # outputs = model(images)
        # loss = loss_fn(outputs, labels)
        # _, pred = outputs.max(1)

    x = torch.randn(1, 3, 512, 612)
    torch.onnx.export(model, x, 'model.onnx', input_names=['input'], output_names=['output'])
    print('export success')

# 同一个图表显示多条曲线，参考：https://blog.csdn.net/weixin_40261309/article/details/105494274