import torch

from core.data_manager.feature_data.RMSFeatureExtractor import RMSFeatureExtractor
from core.data_manager.raw_data.XJTUDataLoader import XJTUDataLoader
from core.data_manager.train_data.SlideWindowDataGenerator import SlideWindowDataGenerator
from core.entity.Bearing import Bearing
from core.model.PytorchModel import PytorchModel
from core.model.mlp.MLP_64_48_32 import MLP_64_48_32
from core.stage_calculator.BearingStageCalculator import BearingStageCalculator
from core.stage_calculator.eol_calculator.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from core.stage_calculator.eol_calculator.TenMaxAmplitudeEoLCalculator import TenMaxAmplitudeEoLCalculator
from core.stage_calculator.eol_calculator.EightMeanRMSEoLCalculator import EightMeanRMSEoLCalculator
from core.stage_calculator.fpt_calculator.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

if __name__ == '__main__':
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyFivePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 32768)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader.get_bearing("Bearing1_3", column='Horizontal Vibration')
    bearing.feature_data = feature_extractor.extract(bearing.raw_data)
    stage_calculator.calculate_state(bearing)
    bearing.plot_feature()

    # 生成训练数据
    data_generator = SlideWindowDataGenerator(96)
    bearing.train_data = data_generator.generate_data(bearing.feature_data)
    print(bearing.train_data)

    # 定义模型并训练
    model = PytorchModel(MLP_64_48_32())
    model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 100)
    model.plot_loss()

    # 使用模型进行预测
    result = model.predict(torch.tensor(bearing.train_data.iloc[20].values[:-32], dtype=torch.float64))
    print(result)
