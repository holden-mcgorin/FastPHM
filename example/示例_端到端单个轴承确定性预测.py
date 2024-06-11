from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.dataset.RelativeRULGenerator import RelativeRULGenerator
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.mlp.FcReluFcRelu import FcReluFcRelu
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.predict.evaluator.End2EndEvaluator import End2EndEvaluator
from rulframework.predict.evaluator.end2end_metric.End2EndMSE import End2EndMSE
from rulframework.predict.evaluator.end2end_metric.End2EndRMSE import End2EndRMSE
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

    # 生成训练数据
    data_generator = RelativeRULGenerator(128)
    data_set = data_generator.generate(bearing)
    train_set, test_set = data_set.split(0.7)

    # 通过其他轴承增加训练数据
    bearing1_1 = data_loader.get_bearing('Bearing1_1')
    bearing1_1.feature_data = feature_extractor.extract(bearing1_1.raw_data)
    stage_calculator.calculate_state(bearing1_1)
    train_set_1_1 = data_generator.generate(bearing1_1)
    train_set.append(train_set_1_1)

    # 定义模型并训练
    model = PytorchModel(FcReluFcRelu([128, 64, 1]))
    model.end2end_train(train_set, 10, weight_decay=0.01)
    Plotter.loss(model)

    result = model.end2end_predict(test_set)
    Plotter.end2end_rul(test_set, result, bearing)

    # 预测结果评价
    evaluator = End2EndEvaluator()
    evaluator.add_metric(End2EndRMSE(), End2EndMSE())
    evaluator.evaluate(test_set, result)
