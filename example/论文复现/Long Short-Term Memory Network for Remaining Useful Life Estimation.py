from torch import nn

from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.loader.bearing.XJTULoader import XJTULoader
from rulframework.data.labeler.RulLabeler import RulLabeler
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.model.pytorch.PytorchModel import PytorchModel
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.metric.Evaluator import Evaluator
from rulframework.metric.end2end.MAE import MAE
from rulframework.metric.end2end.MSE import MSE
from rulframework.metric.end2end.RMSE import RMSE
from rulframework.util.Plotter import Plotter


class ProposedModel(nn.Module):
    def __init__(self, input_size):
        super(ProposedModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=14, kernel_size=3)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(((input_size - 4) // 2 - 2) // 2 * 14, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # use_cache = True

    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader("Bearing1_3", 'Horizontal Vibration')
    feature_extractor(bearing)
    stage_calculator(bearing)

    # 生成训练数据
    generator = RulLabeler(2048, is_from_fpt=False, is_rectified=True)
    data_set = generator(bearing)
    train_set, test_set = data_set.split(0.7)
    # train_set.clear()

    # # 通过其他轴承增加训练数据
    # for bearing_name in ['Bearing1_1', 'Bearing1_2',
    #                      'Bearing2_3', 'Bearing2_2', 'Bearing2_4', 'Bearing2_5',
    #                      'Bearing3_3']:
    #     bearing_train = data_loader(bearing_name)
    #     feature_extractor(bearing_train)
    #     stage_calculator(bearing_train)
    #     another_dataset = generator(bearing_train)
    #     another_dataset, _ = another_dataset.split(0.7)
    #     train_set.append(another_dataset)
    #     # test_set.append(_)

    # train_set, test_set_1 = data_set.split(0.7)

    # 定义模型并训练
    model = PytorchModel(ProposedModel(2048))
    model.train(train_set, 10)
    Plotter.loss(model)

    result = model.test(test_set)
    Plotter.rul_end2end(test_set, result, is_scatter=False)

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE())
    evaluator(test_set, result)
