from torch import nn
import torch.nn.functional as F


class CnnBackbone(nn.Module):
    def __init__(self):
        super(CnnBackbone, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=16)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=0)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x
