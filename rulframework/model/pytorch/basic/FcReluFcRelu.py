from torch import nn


class FcReluFcRelu(nn.Module):
    def __init__(self, size_list: list):
        super(FcReluFcRelu, self).__init__()
        self.fc1 = nn.Linear(size_list[0], size_list[1])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(size_list[1], size_list[2])

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
