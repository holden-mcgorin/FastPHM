from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.train.SlideWindowDataGenerator import SlideWindowDataGenerator
from rulframework.entity.Bearing import PredictHistory
from rulframework.evaluator.Evaluator import Evaluator
from rulframework.evaluator.metric.CI import CI
from rulframework.evaluator.metric.Error import Error
from rulframework.evaluator.metric.ErrorPercentage import ErrorPercentage
from rulframework.evaluator.metric.MAPE import MAPE
from rulframework.evaluator.metric.MSE import MSE
from rulframework.evaluator.metric.Median import Median
from rulframework.evaluator.metric.RUL import RUL
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.uncertainty.MLP_60_48_drop_32 import MLP_60_48_drop_32
from rulframework.predictor.RollingPredictor import RollingPredictor
from rulframework.predictor.confidence_interval.MeanPlusStdCICalculator import MeanPlusStdCICalculator
from rulframework.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.stage.eol.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from rulframework.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.utils.MovingAverageFilter import MovingAverageFilter
from rulframework.utils.ThresholdTrimmer import ThresholdTrimmer

if __name__ == '__main__':
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyFivePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 32768)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader.get_bearing("Bearing1_3", 'Horizontal Vibration')
    bearing.feature_data = feature_extractor.extract(bearing.raw_data)
    stage_calculator.calculate_state(bearing)

    # 生成训练数据
    data_generator = SlideWindowDataGenerator(92)  # 输入大小60+输出大小32=92
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = PytorchModel(MLP_60_48_drop_32())
    model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 100)
    model.plot_loss()

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    ci_calculator = MeanPlusStdCICalculator(1.5)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[:60]
    min_list, mean_list, max_list = predictor.predict_till_epoch_uncertainty(input_data, 5, ci_calculator)

    # 使用移动平均滤波器平滑预测结果
    average_filter = MovingAverageFilter(5)
    min_list = average_filter.moving_average(min_list)
    mean_list = average_filter.moving_average(mean_list)
    max_list = average_filter.moving_average(max_list)

    # 裁剪超过阈值部分曲线
    predict_history = PredictHistory(58, min_list=min_list, mean_list=mean_list, max_list=max_list)
    trimmer = ThresholdTrimmer(bearing.stage_data.failure_threshold_feature)
    bearing.predict_history = trimmer.trim(predict_history)

    bearing.plot_feature()

    # 计算评价指标
    evaluator = Evaluator()
    evaluator.add_metric(RUL(), Median(), CI(), Error(), ErrorPercentage(), MSE(), MAPE())
    evaluator.evaluate(bearing)
