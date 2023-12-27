import torch.optim
import torchvision.datasets
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_mnist_train = torchvision.datasets.MNIST("./mnist", train=True, transform=torchvision.transforms.ToTensor(),
                                                 download=True)
dataset_mnist_test = torchvision.datasets.MNIST("./mnist", train=False, transform=torchvision.transforms.ToTensor())

print(len(dataset_mnist_train))
print(len(dataset_mnist_test))

dataloader_train = DataLoader(dataset_mnist_train, 128, shuffle=True, num_workers=0)

model = Net()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
# opt = torch.optim.SGD(model.parameters(), lr=0.01)
opt = torch.optim.RMSprop(model.parameters(), lr=0.001)

eval_loss = 0
eval_acc = 0
for epoch in range(100):
    for data in dataloader_train:
        imgs, targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        # print(outputs)
        loss = loss_fn(outputs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # print(loss.item())
        eval_loss += loss.item()
        # num_correct = (outputs == targets).sum().item()
        pass
        # 目标label是0-9，预测label是张量，是否需要将目标label归一化
        # 神经网络只考虑单张图像的情况，多张图像由框架自动适应。

    print(f"训练次数：{epoch}, 损失：{eval_loss}")
