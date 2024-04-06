from pandas import DataFrame

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
from rulframework.model.uncertainty.MLP_60_48_drop_32 import MLP_60_48_drop_32
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
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 32768)

    # 划分训练集与测试集（全部是外圈故障轴承）（外圈退化较怕平稳，内圈退化较急促）
    train_set = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing2_2']
    test_set = ['Bearing2_5']

    # 定义模型
    model = PytorchModel(MLP_60_48_drop_32())

    # 合并训练数据
    data_generator = SlideWindowDataGenerator(92)
    train_data_x, train_data_y = DataFrame(), DataFrame()
    for train_bearing in train_set:
        print(f'正在使用{train_bearing}构造训练数据...')
        bearing = data_loader.get_bearing(train_bearing, 'Horizontal Vibration')
        bearing.feature_data = feature_extractor.extract(bearing.raw_data)
        bearing.train_data = data_generator.generate_data(bearing.feature_data)
        train_data_x = train_data_x.append(bearing.train_data.iloc[:, :-32], ignore_index=True)
        train_data_y = train_data_y.append(bearing.train_data.iloc[:, -32:], ignore_index=True)

    # 训练模型
    print('开始训练模型...')
    model.train(train_data_x, train_data_y, 1000, weight_decay=0.1)
    model.plot_loss()

    # 使用测试集预测
    ci_calculator = MeanPlusStdCICalculator(1.5)
    predictor = RollingPredictor(model)
    for teat_bearing in test_set:
        print(f'正在使用{teat_bearing}测试模型...')
        bearing = data_loader.get_bearing(teat_bearing, 'Horizontal Vibration')
        bearing.feature_data = feature_extractor.extract(bearing.raw_data)
        stage_calculator.calculate_state(bearing)
        fpt = bearing.stage_data.fpt_feature
        input_data = bearing.feature_data.iloc[:, 0].tolist()[fpt - 60:fpt]
        lower, prediction, upper = predictor.predict_till_epoch_uncertainty(input_data, 8, ci_calculator)

        # 使用移动平均滤波器平滑预测结果
        average_filter = MovingAverageFilter(10)
        lower, prediction, upper = average_filter.moving_average(lower, prediction, upper)

        # 裁剪超过阈值部分曲线
        predict_history = PredictHistory(fpt, lower=lower, prediction=prediction, upper=upper)
        trimmer = ThresholdTrimmer(bearing.stage_data.failure_threshold_feature)
        bearing.predict_history = trimmer.trim(predict_history)

        # 计算评价指标
        evaluator = Evaluator()
        evaluator.add_metric(RUL(), Mean(), CI(), Error(), ErrorPercentage(), MSE(), MAPE())
        evaluator.evaluate(bearing)

        bearing.plot_feature()
