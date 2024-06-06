from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.train.SlideWindowDataGenerator import SlideWindowDataGenerator
from rulframework.entity.Bearing import PredictHistory
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.mlp.MLP_fc_relu_fc import MLP_fc_relu_fc
from rulframework.predict.ThresholdTrimmer import ThresholdTrimmer
from rulframework.predict.evaluator.Evaluator import Evaluator
from rulframework.predict.evaluator.metric.Error import Error
from rulframework.predict.evaluator.metric.ErrorPercentage import ErrorPercentage
from rulframework.predict.evaluator.metric.MAPE import MAPE
from rulframework.predict.evaluator.metric.MSE import MSE
from rulframework.predict.evaluator.metric.Mean import Mean
from rulframework.predict.evaluator.metric.RUL import RUL
from rulframework.predict.predictor.RollingPredictor import RollingPredictor
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

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
    bearing.plot_feature()

    # 生成训练数据
    data_generator = SlideWindowDataGenerator(92)
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = PytorchModel(MLP_fc_relu_fc([60, 48, 32]))
    model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 1000, weight_decay=0.01)
    model.plot_loss()

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[0:60]
    prediction = predictor.predict_till_threshold(input_data, bearing.stage_data.failure_threshold_feature)
    # prediction = predictor.predict_till_epoch(input_data, 1000)

    # 裁剪超过阈值部分曲线
    predict_history = PredictHistory(59, prediction=prediction)
    trimmer = ThresholdTrimmer(bearing.stage_data.failure_threshold_feature)
    bearing.predict_history = trimmer.trim(predict_history)

    bearing.plot_feature()

    # 计算评价指标
    evaluator = Evaluator()
    evaluator.add_metric(RUL(), Mean(), Error(), ErrorPercentage(), MSE(), MAPE())
    evaluator.evaluate(bearing)
