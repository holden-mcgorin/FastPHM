from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.train.SlideWindowDataGenerator import SlideWindowDataGenerator
from rulframework.entity.Bearing import PredictHistory
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.mlp.MLP_60_48_32 import MLP_60_48_32
from rulframework.predictor.RollingPredictor import RollingPredictor
from rulframework.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

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
    model = PytorchModel(MLP_60_48_32())

    # 使用训练集训练模型
    data_generator = SlideWindowDataGenerator(92)
    for train_bearing in train_set:
        print(f'正在使用{train_bearing}训练模型：')
        bearing = data_loader.get_bearing(train_bearing, 'Horizontal Vibration')
        bearing.feature_data = feature_extractor.extract(bearing.raw_data)
        bearing.train_data = data_generator.generate_data(bearing.feature_data)
        model.train(bearing.train_data.iloc[:, :-32], bearing.train_data.iloc[:, -32:], 10000)
    model.plot_loss()

    # 使用测试集预测
    predictor = RollingPredictor(model)
    for teat_bearing in test_set:
        print(f'正在使用{teat_bearing}测试模型：')
        bearing = data_loader.get_bearing(teat_bearing, 'Horizontal Vibration')
        bearing.feature_data = feature_extractor.extract(bearing.raw_data)
        stage_calculator.calculate_state(bearing)
        fpt = bearing.stage_data.fpt_feature
        input_data = bearing.feature_data.iloc[:, 0].tolist()[fpt - 60:fpt]
        prediction = predictor.predict_till_threshold(input_data, bearing.stage_data.failure_threshold_feature)
        bearing.predict_history = PredictHistory(fpt, prediction)
        bearing.plot_feature()
