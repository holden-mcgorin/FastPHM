from torch import nn
import torch.nn.functional as F


class FcReluFcSoftmax(nn.Module):
    def __init__(self, size_list: list):
        super(FcReluFcSoftmax, self).__init__()
        self.fc1 = nn.Linear(size_list[0], size_list[1])
        self.fc2 = nn.Linear(size_list[1], size_list[2])

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
