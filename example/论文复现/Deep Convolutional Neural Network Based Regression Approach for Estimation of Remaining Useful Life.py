import numpy as np
import torch
from torch import nn

from rulframework.data.Dataset import Dataset
from rulframework.data.FeatureExtractorStream import FeatureExtractorStream
from rulframework.data.labeler.RulTurbofanLabeler import RulTurbofanLabeler
from rulframework.data.loader.turbofan.CMAPSSLoader import CMAPSSLoader
from rulframework.data.processor.NormalizationProcessor import NormalizationProcessor
from rulframework.entity.Turbofan import Turbofan
from rulframework.metric.Evaluator import Evaluator
from rulframework.metric.degeneration.MSE import MSE
from rulframework.metric.end2end.MAE import MAE
from rulframework.metric.end2end.NASAScore import NASAScore
from rulframework.metric.end2end.PHM2012Score import PHM2012Score
from rulframework.metric.end2end.PercentError import PercentError
from rulframework.metric.end2end.RMSE import RMSE
from rulframework.model.pytorch.PytorchModel import PytorchModel
from rulframework.util.Plotter import Plotter

"""
首篇论文将CNN应用于RUL预测
数据集：CMAPSS、PHM2008
模型：CNN（2维卷积）
输入：矩阵，使用滑动窗口提取（窗口大小：15、步长：1）
输出（RUL）：分段线性RUL（piece-wise linear degradation model），根据窗口最后一行计算RUL
损失函数：均方误差
优化算法：随机梯度下降
训练参数：

"""


class ProposedModel(nn.Module):
    def __init__(self, feature_size):
        super(ProposedModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(feature_size, 4))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=14, kernel_size=(1, 3))
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


columns_to_drop = [0, 1, 2, 3, 4, 8, 9, 13, 19, 21, 22]
# columns_to_drop = [0, 1, 2, 3]
# feature_size = 21
feature_size = 14


def batch_load(trajectory: str) -> [Turbofan]:
    data_loader = CMAPSSLoader('D:\\data\\dataset\\CMAPSSData')

    entity_name_list = []
    entities = []

    for i in range(1, CMAPSSLoader.trajectories[trajectory] + 1):
        entity_name_list.append(trajectory + '_' + str(i))

    for name in entity_name_list:
        entities.append(data_loader(name, columns_to_drop))

    t_min = np.delete(data_loader.arr_min[trajectory], columns_to_drop)
    t_max = np.delete(data_loader.arr_max[trajectory], columns_to_drop)
    p = NormalizationProcessor(arr_min=t_min, arr_max=t_max)
    feature_extractor = FeatureExtractorStream([p])

    for entity in entities:
        feature_extractor(entity)
        # entity.feature_data = entity.raw_data

    return entities


if __name__ == '__main__':
    labeler = RulTurbofanLabeler(window_size=15, window_step=1, max_rul=125)

    turbofans_train = batch_load('FD001_train')
    train_set = Dataset()
    for turbofan in turbofans_train:
        train_set.append(labeler(turbofan))

    turbofan_test = batch_load('FD001_test')
    test_set = Dataset()
    for turbofan in turbofan_test:
        test_set.append(labeler(turbofan))

    # 定义模型并训练
    model = PytorchModel(ProposedModel(feature_size))
    model.train(train_set, epochs=120, batch_size=256, lr=0.01)
    Plotter.loss(model)

    # 做出预测并画预测结果
    result = model.test(test_set)
    Plotter.rul_end2end(test_set, result, is_scatter=True)

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), NASAScore())
    evaluator(test_set, result)
