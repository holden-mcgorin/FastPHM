import torch
from torch import nn


class CMAPSSModel(nn.Module):
    def __init__(self):
        super(CMAPSSModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(96, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度，x的形状(批量，通道，传感器，时间)
        shortcut1 = x
        shortcut1 = self.conv1_1(shortcut1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = x + shortcut1
        x = self.pool(x)
        shortcut2 = x
        shortcut2 = self.conv2_1(shortcut2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x + shortcut2
        x = self.pool(x)
        shortcut3 = x
        x = self.conv3(x)
        x = torch.relu(x)
        x = x + shortcut3
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = 125 * torch.sigmoid(x)
        return x
