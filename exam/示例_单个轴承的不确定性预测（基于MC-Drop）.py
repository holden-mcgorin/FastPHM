from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.train.SlideWindowDataGenerator import SlideWindowDataGenerator
from rulframework.entity.Bearing import PredictHistory
from rulframework.predict.evaluator.Evaluator import Evaluator
from rulframework.predict.evaluator.metric.CI import CI
from rulframework.predict.evaluator.metric.Error import Error
from rulframework.predict.evaluator.metric.ErrorPercentage import ErrorPercentage
from rulframework.predict.evaluator.metric.MAPE import MAPE
from rulframework.predict.evaluator.metric.MSE import MSE
from rulframework.predict.evaluator.metric.Mean import Mean
from rulframework.predict.evaluator.metric.RUL import RUL
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.uncertainty.MLP_fc_drop_fc_relu import MLP_fc_drop_fc_relu
from rulframework.predict.predictor.RollingPredictor import RollingPredictor
from rulframework.predict.confidence_interval.MeanPlusStdCICalculator import MeanPlusStdCICalculator
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.util.MovingAverageFilter import MovingAverageFilter
from rulframework.predict.ThresholdTrimmer import ThresholdTrimmer

if __name__ == '__main__':
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    # data_loader = PHM2012DataLoader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    feature_extractor = RMSFeatureExtractor(data_loader.span)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.span)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader.get_bearing("Bearing1_3", 'Horizontal Vibration')
    bearing.feature_data = feature_extractor.extract(bearing.raw_data)
    stage_calculator.calculate_state(bearing)
    bearing.plot_feature()

    # 生成训练数据
    data_generator = SlideWindowDataGenerator(92)  # 输入大小60+输出大小32=92
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = PytorchModel(MLP_fc_drop_fc_relu([60, 48, 32]))
    model.train(bearing.train_data[:, :-32], bearing.train_data[:, -32:], 100, weight_decay=0)
    model.plot_loss()

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    ci_calculator = MeanPlusStdCICalculator(1.5)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[:60]
    lower, prediction, upper = predictor.predict_till_epoch_uncertainty(input_data, 4, ci_calculator)

    # 使用移动平均滤波器平滑预测结果
    average_filter = MovingAverageFilter(5)
    lower, prediction, upper = average_filter.moving_average(lower, prediction, upper)

    # 裁剪超过阈值部分曲线
    predict_history = PredictHistory(59, lower=lower, prediction=prediction, upper=upper)
    trimmer = ThresholdTrimmer(bearing.stage_data.failure_threshold_feature)
    bearing.predict_history = trimmer.trim(predict_history)

    bearing.plot_feature()

    # 计算评价指标
    evaluator = Evaluator()
    evaluator.add_metric(RUL(), Mean(), CI(), Error(), ErrorPercentage(), MSE(), MAPE())
    evaluator.evaluate(bearing)
