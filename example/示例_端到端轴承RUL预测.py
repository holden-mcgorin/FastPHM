from fastphm.data.FeatureExtractor import FeatureExtractor
from fastphm.data.loader.bearing.XJTULoader import XJTULoader
from fastphm.data.labeler.RulLabeler import RulLabeler
from fastphm.data.processor.RMSProcessor import RMSProcessor
from fastphm.metric.end2end.PHM2008Score import PHM2008Score
from fastphm.metric.end2end.PHM2012Score import PHM2012Score
from fastphm.metric.end2end.PercentError import PercentError
from fastphm.model.pytorch.PytorchModel import PytorchModel
from fastphm.data.stage.BearingStageCalculator import BearingStageCalculator
from fastphm.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from fastphm.model.pytorch.basic.CNN import CNN
from fastphm.metric.Evaluator import Evaluator
from fastphm.metric.end2end.MAE import MAE
from fastphm.metric.end2end.MSE import MSE
from fastphm.metric.end2end.RMSE import RMSE
from fastphm.util.Cache import Cache
from fastphm.util.Plotter import Plotter

if __name__ == '__main__':
    use_cache = True
    # use_cache = False
    train_set = Cache.load('prognosis_train', is_able=use_cache)
    test_set = Cache.load('prognosis_test', is_able=use_cache)

    if train_set is None or test_set is None:
        # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
        data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
        feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
        fpt_calculator = ThreeSigmaFPTCalculator()
        stage_calculator = BearingStageCalculator(data_loader.continuum, fpt_calculator)

        # 获取原始数据、特征数据、阶段数据
        bearing = data_loader("Bearing1_3", 'Horizontal Vibration')
        feature_extractor(bearing)
        stage_calculator(bearing)

        # 生成训练数据
        generator = RulLabeler(2048, time_ratio=60, is_from_fpt=False, is_rectified=True)
        data_set = generator(bearing)
        train_set, test_set = data_set.split(0.7)
        train_set.clear()

        # 通过其他轴承增加训练数据
        for bearing_name in ['Bearing1_1', 'Bearing1_2']:
            bearing_train = data_loader(bearing_name)
            feature_extractor(bearing_train)
            stage_calculator(bearing_train)
            another_dataset = generator(bearing_train)
            train_set.append(another_dataset)

        Cache.save(train_set, 'prognosis_train')
        Cache.save(test_set, 'prognosis_test')

    # 定义模型并训练
    model = PytorchModel(CNN(2048, 1))
    model.train(train_set, 100)
    Plotter.loss(model)

    result = model.test(test_set)
    Plotter.rul_end2end(test_set, result, is_scatter=True, label_x='Time (min)', label_y='Relative RUL')

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())
    evaluator(test_set, result)
