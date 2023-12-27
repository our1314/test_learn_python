import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from Data import TestData
from Net import Net


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = TestData('D:/desktop/ccc')
    train_loader = DataLoader(train_data, batch_size=1)

    val_data = TestData('D:/desktop/ccc')
    val_loader = DataLoader(val_data, batch_size=1)

    model = Net()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    loss_fn = nn.MSELoss()

    for epoch in range(900):
        loss_value = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets.view(-1, 6))
            loss_value += loss
            loss.backward()
            optimizer.step()

        print(f'loss:{loss_value}')


if __name__ == '__main__':
    train()
