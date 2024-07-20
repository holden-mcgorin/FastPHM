from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=8)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size - 1184, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
