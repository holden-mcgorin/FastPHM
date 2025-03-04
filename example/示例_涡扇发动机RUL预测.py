import torch
from torch import nn

from fastphm.data.Dataset import Dataset
from fastphm.data.labeler.RulTurbofanLabeler import RulTurbofanLabeler
from fastphm.data.loader.turbofan.CMAPSSLoader import CMAPSSLoader
from fastphm.metric.Evaluator import Evaluator
from fastphm.metric.end2end.MSE import MSE
from fastphm.metric.end2end.MAE import MAE
from fastphm.metric.end2end.PHM2008Score import PHM2008Score
from fastphm.metric.end2end.PHM2012Score import PHM2012Score
from fastphm.metric.end2end.PercentError import PercentError
from fastphm.metric.end2end.RMSE import RMSE
from fastphm.model.pytorch.PytorchModel import PytorchModel
from fastphm.util.Plotter import Plotter


class ProposedModel(nn.Module):
    def __init__(self):
        super(ProposedModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

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


data_loader = CMAPSSLoader('D:\\data\\dataset\\CMAPSSData')
labeler_all_sample = RulTurbofanLabeler(window_size=15, window_step=1, max_rul=125)
labeler = RulTurbofanLabeler(window_size=15, window_step=1, max_rul=125, last_cycle=True)
columns_to_drop = [0, 1, 2, 3, 4, 8, 9, 13, 19, 21, 22]
feature_size = 14
Plotter.DPI = 80

turbofan = data_loader('FD001_train_1', columns_to_drop)
Plotter.raw(turbofan)
Plotter.feature(turbofan)

turbofans_train = data_loader.batch_load('FD001_train', columns_to_drop)
train_set = Dataset()
for turbofan in turbofans_train:
    train_set.append(labeler_all_sample(turbofan))

turbofan_test = data_loader.batch_load('FD001_test', columns_to_drop)
test_set = Dataset()
for turbofan in turbofan_test:
    test_set.append(labeler(turbofan))

model = PytorchModel(ProposedModel())
model.train(train_set, epochs=120, batch_size=256, lr=0.01, weight_decay=0.01)
Plotter.loss(model)

result = model.test(test_set)

test_set.name = 'FD001_test'
evaluator = Evaluator()
evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())
evaluator(test_set, result)
Plotter.rul_ascending(test_set, result, is_scatter=False, label_x='Test Engine ID', label_y='RUL (cycle)')

turbofan = data_loader('FD001_test_100', columns_to_drop)
a_test = labeler_all_sample(turbofan)
result_all_sample = model.test(a_test)
Plotter.rul_end2end(a_test, result_all_sample, is_scatter=False, label_x='Time (cycle)', label_y='RUL (cycle)')
