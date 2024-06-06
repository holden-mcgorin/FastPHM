from torch import nn


class MLP_fc_drop_fc_relu(nn.Module):
    def __init__(self, size_list: list):
        super(MLP_fc_drop_fc_relu, self).__init__()
        self.fc1 = nn.Linear(size_list[0], size_list[1])
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(size_list[1], size_list[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
