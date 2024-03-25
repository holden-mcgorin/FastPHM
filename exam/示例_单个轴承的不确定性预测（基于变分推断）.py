from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.train.SlideWindowDataGenerator import SlideWindowDataGenerator
from rulframework.entity.Bearing import PredictHistory
from rulframework.model.BnnModel import BnnModel
from rulframework.model.uncertainty.BNN_60_48_32 import BNN_60_48_32
from rulframework.predictor.RollingPredictor import RollingPredictor
from rulframework.predictor.confidence_interval.MeanPlusStdCICalculator import MeanPlusStdCICalculator
from rulframework.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.stage.eol.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from rulframework.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

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
    bearing.plot_feature()

    # 生成训练数据
    data_generator = SlideWindowDataGenerator(92)  # 输入大小60+输出大小32=92
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = BnnModel(BNN_60_48_32())
    model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 100)
    model.plot_loss()

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    ci_calculator = MeanPlusStdCICalculator(2)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[:60]
    min_list, mean_list, max_list = \
        predictor.predict_till_epoch_uncertainty_flat(input_data, 3, bearing.stage_data.failure_threshold_feature,
                                                      ci_calculator)

    bearing.predict_history = PredictHistory(59, min_list=min_list, mean_list=mean_list, max_list=max_list)
    bearing.plot_feature()
