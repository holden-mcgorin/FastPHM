from torch import nn

from fastphm.data.Dataset import Dataset
from fastphm.data.FeatureExtractor import FeatureExtractor
from fastphm.data.labeler.FaultLabeler import FaultLabeler
from fastphm.data.loader.bearing.XJTULoader import XJTULoader
from fastphm.data.processor.RMSProcessor import RMSProcessor
from fastphm.entity.Bearing import Fault
from fastphm.model.pytorch.PytorchModel import PytorchModel
from fastphm.data.stage.BearingStageCalculator import BearingStageCalculator
from fastphm.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from fastphm.model.pytorch.basic.CNN import CNN
from fastphm.util.Cache import Cache
from fastphm.util.Plotter import Plotter

fault_types = [Fault.NC, Fault.OF, Fault.IF, Fault.CF]

if __name__ == '__main__':
    use_cache = True
    # use_cache = False
    train_set = Cache.load('diagnosis_train', is_able=use_cache)
    test_set = Cache.load('diagnosis_test', is_able=use_cache)

    if train_set is None or test_set is None:
        # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
        data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
        feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
        fpt_calculator = ThreeSigmaFPTCalculator()
        stage_calculator = BearingStageCalculator(data_loader.continuum, fpt_calculator)

        # 生成训练数据
        dataset = Dataset()
        generator = FaultLabeler(2048, fault_types, is_onehot=False)
        for bearing_name in ['Bearing1_1', "Bearing1_3",
                             'Bearing2_3', 'Bearing2_4',
                             'Bearing3_3']:
            bearing_train = data_loader(bearing_name, 'Horizontal Vibration')
            feature_extractor(bearing_train)
            stage_calculator(bearing_train)
            another_dataset = generator(bearing_train)
            dataset.append(another_dataset)

        train_set, test_set = dataset.split(0.7)

        Cache.save(train_set, 'diagnosis_train')
        Cache.save(test_set, 'diagnosis_test')

    # 定义模型并训练
    model = PytorchModel(CNN(2048, len(fault_types)))
    model.train(train_set, 100, criterion=nn.CrossEntropyLoss())
    Plotter.loss(model)

    result = model.test(test_set)
    Plotter.diagnosis_confusion_matrix(test_set, result, types=fault_types)
