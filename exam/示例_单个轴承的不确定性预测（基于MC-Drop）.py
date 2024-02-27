from core.data_manager.feature_data.RMSFeatureExtractor import RMSFeatureExtractor
from core.data_manager.raw_data.XJTUDataLoader import XJTUDataLoader
from core.data_manager.train_data.SlideWindowDataGenerator import SlideWindowDataGenerator
from core.entity.Bearing import PredictHistory
from core.model.PytorchModel import PytorchModel
from core.model.uncertainty.MLP_64_48_drop_32 import MLP_64_48_drop_32
from core.predictor.RollingPredictor import RollingPredictor
from core.stage_calculator.BearingStageCalculator import BearingStageCalculator
from core.stage_calculator.eol_calculator.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from core.stage_calculator.fpt_calculator.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

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
    data_generator = SlideWindowDataGenerator(96)  # 输入大小64+输出大小32=96
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = PytorchModel(MLP_64_48_drop_32())
    model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 100)
    model.plot_loss()

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[:64]
    min_list, mean_list, max_list = predictor.predict_till_epoch_uncertainty(input_data, 3)

    bearing.predict_history = PredictHistory(63, min_list=min_list, mean_list=mean_list, max_list=max_list)
    bearing.plot_feature()
