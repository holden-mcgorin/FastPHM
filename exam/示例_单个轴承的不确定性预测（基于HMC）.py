from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.train.SlideWindowDataGenerator import SlideWindowDataGenerator
from rulframework.entity.Bearing import Result
from rulframework.model.PyroModel import PyroModel
from rulframework.model.mlp.FcReluFcRelu import FcReluFcRelu
from rulframework.predict.predictor.RollingPredictor import RollingPredictor
from rulframework.predict.confidence_interval.MeanPlusStdCICalculator import MeanPlusStdCICalculator
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
    data_generator = SlideWindowDataGenerator(92)  # 输入大小60+输出大小32=92
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = PyroModel(FcReluFcRelu([60, 48, 32]))
    model.train(bearing.train_data[:, :-32], bearing.train_data[:, -32:], 100)

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    ci_calculator = MeanPlusStdCICalculator(2)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[:60]
    min_list, mean_list, max_list = \
        predictor.predict_till_epoch_uncertainty_flat(input_data, 4, bearing.stage_data.failure_threshold_feature,
                                                      ci_calculator)

    bearing.result = Result(59, upper=max_list, lower=min_list)
    Plotter.degeneration(bearing)
