from rulframework.data_manager.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data_manager.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data_manager.train.SlideWindowDataGenerator import SlideWindowDataGenerator
from rulframework.entity.Bearing import PredictHistory
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.uncertainty.MLP_60_48_drop_32 import MLP_60_48_drop_32
from rulframework.predictor.RollingPredictor import RollingPredictor
from rulframework.predictor.confidence_interval.MeanPlusStdCICalculator import MeanPlusStdCICalculator
from rulframework.predictor.confidence_interval.MiddleSampleCICalculator import MiddleSampleCICalculator
from rulframework.stage_calculator.BearingStageCalculator import BearingStageCalculator
from rulframework.stage_calculator.eol.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from rulframework.stage_calculator.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

if __name__ == '__main__':
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyFivePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 32768)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader.get_bearing("Bearing1_3", column='Horizontal Vibration')
    bearing.feature_data = feature_extractor.extract(bearing.raw_data)
    stage_calculator.calculate_state(bearing)
    bearing.plot_feature()

    # 生成训练数据
    data_generator = SlideWindowDataGenerator(92)  # 输入大小60+输出大小32=92
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = PytorchModel(MLP_60_48_drop_32())
    model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 100)
    model.plot_loss()

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    ci_calculator = MeanPlusStdCICalculator(1)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[:60]
    min_list, mean_list, max_list = \
        predictor.predict_till_epoch_uncertainty_flat(input_data, 5, bearing.stage_data.failure_threshold_feature,
                                                      ci_calculator)

    bearing.predict_history = PredictHistory(59, min_list=min_list, mean_list=mean_list, max_list=max_list)
    bearing.plot_feature()
