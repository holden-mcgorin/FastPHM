from core.data_manager.feature_data.RMSFeatureExtractor import RMSFeatureExtractor
from core.data_manager.raw_data.XJTUDataLoader import XJTUDataLoader
from core.data_manager.train_data.SlideWindowDataGenerator import SlideWindowDataGenerator
from core.entity.Bearing import PredictHistory
from core.model.PytorchModel import PytorchModel
from core.model.mlp.MLP_64_48_32 import MLP_64_48_32
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
    data_generator = SlideWindowDataGenerator(96)
    bearing.train_data = data_generator.generate_data(bearing.feature_data)

    # 定义模型并训练
    model = PytorchModel(MLP_64_48_32())
    model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 10000)
    model.plot_loss()

    # 使用预测器进行预测
    predictor = RollingPredictor(model)
    input_data = bearing.feature_data.iloc[:, 0].tolist()[0:64]
    prediction = predictor.predict_till_threshold(input_data, bearing.stage_data.failure_threshold_feature)
    bearing.predict_history = PredictHistory(64, prediction)

    bearing.plot_feature()
