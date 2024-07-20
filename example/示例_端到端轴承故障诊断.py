from torch import nn

from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.label.FaultLabelGenerator import FaultLabelGenerator
from rulframework.data.loader.XJTUDataLoader import XJTUDataLoader
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.entity.Bearing import BearingFault
from rulframework.model.pytorch.PytorchModel import PytorchModel
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.model.pytorch.classic.LeNet import LeNet
from rulframework.util.Plotter import Plotter

if __name__ == '__main__':
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader.get("Bearing1_3", 'Horizontal Vibration')
    feature_extractor.extract(bearing)
    stage_calculator.calculate_state(bearing)

    # 生成训练数据
    generator = FaultLabelGenerator(2048, list(BearingFault.__members__.values()), is_onehot=False)
    dataset = generator.generate(bearing)
    train_set, test_set = dataset.split(0.7)
    # train_set.clear()

    # # 通过其他轴承增加训练数据
    # for bearing_name in ['Bearing1_1', 'Bearing1_4', 'Bearing2_1', 'Bearing1_2', 'Bearing2_3']:
    #     bearing_train = data_loader.get(bearing_name, 'Horizontal Vibration')
    #     feature_extractor.extract(bearing_train)
    #     stage_calculator.calculate_state(bearing_train)
    #     another_dataset = generator.generate(bearing_train)
    #     another_dataset, _ = another_dataset.split(0.7)
    #     train_set.append(another_dataset)
    #     test_set.append(_)

    # train_set, test_set = label.split(0.7)

    # 定义模型并训练
    model = PytorchModel(LeNet(input_length=2048, output_length=5))
    # model = PytorchModel(AlexNet(input_length=2048, output_length=5))
    # model = PytorchModel(ResNet18())
    # model = PytorchModel(CNN(2048, 5))
    # model = PytorchModel(CNN(), criterion=nn.BCELoss())
    # model = PytorchModel(CNN(), criterion=nn.MSELoss())
    # model = PytorchModel(FcReluFcSoftmax([128, 64, 5]), criterion=nn.MSELoss())
    # model = PytorchModel(FcReluFcSoftmax([128, 64, 5]), criterion=nn.CrossEntropyLoss())

    model.train(train_set, 10, weight_decay=0.01, criterion=nn.CrossEntropyLoss())
    Plotter.loss(model)

    result = model.test(test_set)

    Plotter.fault_diagnosis_evolution(test_set, result, interval=1)
    Plotter.fault_diagnosis_heatmap(test_set, result)
