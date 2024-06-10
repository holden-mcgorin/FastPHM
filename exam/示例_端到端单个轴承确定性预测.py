import numpy as np

from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.train.RelativeRUL import RelativeRUL
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.mlp.FcReluFcRelu import FcReluFcRelu
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.util.Plotter import Plotter

if __name__ == '__main__':
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 32768)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader.get_bearing("Bearing1_3", 'Horizontal Vibration')
    bearing.feature_data = feature_extractor.extract(bearing.raw_data)
    stage_calculator.calculate_state(bearing)
    Plotter.feature(bearing)

    # 生成训练数据
    data_generator = RelativeRUL()
    x, y = data_generator.generate(bearing, 128)

    # 定义模型并训练
    model = PytorchModel(FcReluFcRelu([128, 64, 1]))
    model.train(x, y, 100, weight_decay=0.01)
    Plotter.loss(model)


    h_index = np.linspace(0, x.shape[0], x.shape[0])
    v_index = model(x).reshape(-1)

    Plotter.relative_rul(h_index, v_index)
    # # 使用预测器进行预测
    # predictor = RollingPredictor(model)
    # input_data = bearing.feature_data.iloc[:, 0].tolist()[0:60]
    # mean = predictor.predict_till_threshold(input_data, bearing.stage_data.failure_threshold_feature)
    # # mean = predictor.predict_till_epoch(input_data, 1000)
    #
    # # 裁剪超过阈值部分曲线
    # result = Result(59, mean=mean)
    # trimmer = ThresholdTrimmer(bearing.stage_data.failure_threshold_feature)
    # bearing.result = trimmer.trim(result)
    #
    # bearing.plot_feature()
    #
    # # 计算评价指标
    # evaluator = Evaluator()
    # evaluator.add_metric(RUL(), Mean(), Error(), ErrorPercentage(), MSE(), MAPE())
    # evaluator.evaluate(bearing)
