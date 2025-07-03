import torch
from torch import nn

from fastphm.data import Dataset
from fastphm.data.FeatureExtractor import FeatureExtractor
from fastphm.data.labeler.BearingFaultLabeler import BearingFaultLabeler
from fastphm.data.loader.bearing.XJTULoader import XJTULoader
from fastphm.data.processor.RMSProcessor import RMSProcessor
from fastphm.entity.Bearing import Fault
from fastphm.metric.Evaluator import Evaluator
from fastphm.metric.end2end.Accuracy import Accuracy
from fastphm.metric.end2end.WeightedF1Score import WeightedF1Score
from fastphm.data.stage.BearingStageCalculator import BearingStageCalculator
from fastphm.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from fastphm.model.pytorch.base.BaseTester import BaseTester
from fastphm.model.pytorch.base.BaseTrainer import BaseTrainer
from fastphm.model.pytorch.basic.CNN import CNN
from fastphm.model.pytorch.callback.CheckGradientsCallback import CheckGradientsCallback
from fastphm.model.pytorch.callback.EarlyStoppingCallback import EarlyStoppingCallback
from fastphm.model.pytorch.callback.TensorBoardCallback import TensorBoardCallback
from fastphm.system.Cache import Cache
from fastphm.util.Plotter import Plotter

if __name__ == '__main__':
    fault_types = [Fault.NC, Fault.OF, Fault.IF, Fault.CF]

    cache_dataset = True
    # cache_dataset = False

    # cache_model = True
    cache_model = False

    dataset = Cache.load('diagnosis_dataset', is_able=cache_dataset)
    if dataset is None:
        # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
        data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
        feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
        fpt_calculator = ThreeSigmaFPTCalculator()
        stage_calculator = BearingStageCalculator(data_loader.continuum, fpt_calculator)
        generator = BearingFaultLabeler(2048, fault_types, is_multi_hot=False)

        # 通过其他轴承增加训练数据
        dataset = Dataset()
        for bearing_name in ['Bearing1_1', 'Bearing1_2', "Bearing1_3",
                             'Bearing2_3', 'Bearing2_2', 'Bearing2_4', 'Bearing2_5',
                             'Bearing3_3']:
            bearing_train = data_loader(bearing_name, 'Horizontal Vibration')
            feature_extractor(bearing_train)
            stage_calculator(bearing_train)
            dataset.add(generator(bearing_train))
        Cache.save(dataset, 'diagnosis_dataset')

    # 按比例划分训练集和测试集
    train_set, test_set = dataset.split_by_ratio(0.7)

    # 配置测试算法
    tester = BaseTester()

    # 配置训练算法
    train_config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'dtype': torch.float32,
        'epochs': 10,
        'batch_size': 128,
        'lr': 0.001,
        'weight_decay': 0.0,
        'criterion': nn.CrossEntropyLoss(),
        'callbacks': [
            EarlyStoppingCallback(patience=20,
                                  val_set=test_set,
                                  metric=Accuracy(),
                                  tester=tester),
            TensorBoardCallback(),
            CheckGradientsCallback()
        ]
    }
    trainer = BaseTrainer(config=train_config)

    # 定义模型并训练
    model = Cache.load('diagnosis_model', cache_model)
    if model is None:
        model = CNN(2048, len(fault_types))
        # 配置训练算法
        # 开始训练
        losses = trainer.train(model=model, train_set=train_set)
        Plotter.loss(losses)
        Cache.save(model, 'diagnosis_model')

    result = tester.test(model, test_set)
    Plotter.confusion_matrix(test_set, result, types=fault_types)

    Plotter.fault_evolution(test_set.get('Bearing1_1'), result.get('Bearing1_1'), types=fault_types)
    Plotter.fault_evolution(test_set, result, types=fault_types)

    # 预测结果评价（故障诊断）
    evaluator = Evaluator()
    evaluator.add(Accuracy(), WeightedF1Score())
    evaluator(test_set, result)
