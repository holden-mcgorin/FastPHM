from torch import nn


class MLP_64_48_32(nn.Module):
    def __init__(self):
        super(MLP_64_48_32, self).__init__()
        self.fc1 = nn.Linear(64, 48)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(48, 32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
