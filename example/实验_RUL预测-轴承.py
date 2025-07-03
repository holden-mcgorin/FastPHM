import torch

from fastphm.data import Dataset
from fastphm.data.FeatureExtractor import FeatureExtractor
from fastphm.data.loader.bearing.XJTULoader import XJTULoader
from fastphm.data.labeler.BearingRulLabeler import BearingRulLabeler
from fastphm.data.processor.RMSProcessor import RMSProcessor
from fastphm.metric.end2end.PHM2008Score import PHM2008Score
from fastphm.metric.end2end.PHM2012Score import PHM2012Score
from fastphm.metric.end2end.PercentError import PercentError
from fastphm.data.stage.BearingStageCalculator import BearingStageCalculator
from fastphm.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from fastphm.model.pytorch.base.BaseTester import BaseTester
from fastphm.model.pytorch.base.BaseTrainer import BaseTrainer
from fastphm.model.pytorch.basic.CNN import CNN
from fastphm.metric.Evaluator import Evaluator
from fastphm.metric.end2end.MAE import MAE
from fastphm.metric.end2end.MSE import MSE
from fastphm.metric.end2end.RMSE import RMSE
from fastphm.model.pytorch.callback.CheckGradientsCallback import CheckGradientsCallback
from fastphm.model.pytorch.callback.EarlyStoppingCallback import EarlyStoppingCallback
from fastphm.model.pytorch.callback.TensorBoardCallback import TensorBoardCallback
from fastphm.system.Cache import Cache
from fastphm.util.Plotter import Plotter

if __name__ == '__main__':
    cache_dataset = True
    # cache_dataset = False
    # cache_model = True
    cache_model = False

    dataset = Cache.load('prognosis_bearing_dataset', is_able=cache_dataset)
    if dataset is None:
        # 定义 数据加载器、特征提取器、fpt计算器
        data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
        feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
        fpt_calculator = ThreeSigmaFPTCalculator()
        stage_calculator = BearingStageCalculator(data_loader.continuum, fpt_calculator)
        generator = BearingRulLabeler(2048, is_from_fpt=False, is_rectified=True)

        dataset = Dataset()
        for bearing_name in ['Bearing1_1', 'Bearing1_2', 'Bearing1_3',
                             'Bearing2_1', 'Bearing2_2',
                             'Bearing3_1', 'Bearing3_2']:
            bearing_train = data_loader(bearing_name, 'Horizontal Vibration')
            feature_extractor(bearing_train)
            stage_calculator(bearing_train)
            dataset.add(generator(bearing_train))
        Cache.save(dataset, 'prognosis_bearing_dataset')

    # 划分测试集和训练集
    test_set = dataset.get('Bearing1_3')
    train_set = dataset.exclude('Bearing1_3')

    # 配置测试算法
    test_config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'dtype': torch.float32,
    }
    tester = BaseTester(config=test_config)

    # 配置训练算法
    train_config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'dtype': torch.float32,
        'epochs': 20,
        'batch_size': 256,
        'lr': 0.01,
        'weight_decay': 0.0,
        'callbacks': [
            EarlyStoppingCallback(patience=5,
                                  val_set=test_set,
                                  metric=RMSE(),
                                  tester=tester),
            TensorBoardCallback(),
            CheckGradientsCallback()
        ]
    }
    trainer = BaseTrainer(config=train_config)

    # 定义模型并训练
    model = Cache.load('prognosis_bearing_model', cache_model)
    if model is None:
        model = CNN(2048, 1)
        # 开始训练
        losses = trainer.train(model=model, train_set=train_set)
        Plotter.loss(losses)
        Cache.save(model, 'prognosis_bearing_model')

    result = tester.test(model, test_set)
    Plotter.rul_end2end(test_set, result, is_scatter=True, label_x='Time (min)', label_y='Relative RUL')

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())
    evaluator(test_set, result)
