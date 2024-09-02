from torch import nn

from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.labeler.FaultLabeler import FaultLabeler
from rulframework.data.loader.bearing.XJTULoader import XJTULoader
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.entity.Bearing import Fault
from rulframework.model.pytorch.PytorchModel import PytorchModel
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.model.pytorch.basic.CNN import CNN
from rulframework.util.Cache import Cache
from rulframework.util.Plotter import Plotter

fault_types = [Fault.NC, Fault.OF, Fault.IF, Fault.CF]

if __name__ == '__main__':
    # use_cache = True
    use_cache = False
    train_set = Cache.load('diagnosis_train', is_able=use_cache)
    test_set = Cache.load('diagnosis_test', is_able=use_cache)

    if train_set is None or test_set is None:
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
        fault_types = [Fault.NC, Fault.OF, Fault.IF, Fault.CF]
        generator = FaultLabeler(2048, fault_types, is_onehot=False)
        dataset = generator(bearing)
        # train_set, test_set = dataset.split(0.7)
        # train_set.clear()

        # 通过其他轴承增加训练数据
        for bearing_name in ['Bearing1_1', 'Bearing1_2', 'Bearing2_3', 'Bearing2_2', 'Bearing2_4', 'Bearing2_5',
                             'Bearing3_3']:
            bearing_train = data_loader(bearing_name, 'Horizontal Vibration')
            feature_extractor(bearing_train)
            stage_calculator(bearing_train)
            another_dataset = generator(bearing_train)
            dataset.append(another_dataset)
            # another_dataset, _ = another_dataset.split(0.7)
            # train_set.append(another_dataset)
            # test_set.append(_)

        train_set, test_set = dataset.split(0.7)

        Cache.save(train_set, 'diagnosis_train')
        Cache.save(test_set, 'diagnosis_test')

    # 定义模型并训练
    # model = PytorchModel(LeNet(input_length=2048, output_length=4))
    # model = PytorchModel(AlexNet(input_length=2048, output_length=5))
    # model = PytorchModel(ResNet18())
    model = PytorchModel(CNN(2048, len(fault_types)))
    # model = PytorchModel(CNN(), criterion=nn.BCELoss())
    # model = PytorchModel(CNN(), criterion=nn.MSELoss())
    # model = PytorchModel(FcReluFcSoftmax([128, 64, 5]), criterion=nn.MSELoss())
    # model = PytorchModel(FcReluFcSoftmax([128, 64, 5]), criterion=nn.CrossEntropyLoss())

    model.train(train_set, 10, weight_decay=0.01, criterion=nn.CrossEntropyLoss())
    Plotter.loss(model)

    result = model.test(test_set)

    Plotter.fault_diagnosis_evolution(test_set, result, types=fault_types)
    Plotter.fault_diagnosis_heatmap(test_set, result, types=fault_types)
