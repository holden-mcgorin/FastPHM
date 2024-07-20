from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.loader.XJTUDataLoader import XJTUDataLoader
from rulframework.data.label.RulLabelGenerator import RulLabelGenerator
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.model.pytorch.PytorchModel import PytorchModel
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.model.pytorch.basic.CNN import CNN
from rulframework.metric.Evaluator import Evaluator
from rulframework.metric.end2end.End2EndMAE import End2EndMAE
from rulframework.metric.end2end.End2EndMSE import End2EndMSE
from rulframework.metric.end2end.End2EndRMSE import End2EndRMSE
from rulframework.util.Plotter import Plotter

if __name__ == '__main__':
    # 定义 数据加载器、特征提取器、fpt计算器、eol计算器
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    # 获取原始数据、特征数据、阶段数据
    bearing = data_loader.get("Bearing1_3", 'Horizontal Vibration')
    feature_extractor.extract(bearing)
    stage_calculator.calculate_state(bearing)

    # 生成训练数据
    generator = RulLabelGenerator(2048, is_from_fpt=False, is_rectified=True)
    data_set = generator.generate(bearing)
    train_set, test_set = data_set.split(0.7)
    # train_set.clear()
    #
    # # 通过其他轴承增加训练数据
    # for bearing_name in ['Bearing1_1', 'Bearing1_4', 'Bearing2_1', 'Bearing1_2', 'Bearing2_3']:
    #     bearing_train = data_loader.get(bearing_name)
    #     feature_extractor.extract(bearing_train)
    #     stage_calculator.calculate_state(bearing_train)
    #     data_set.append(generator.generate(bearing_train))
    #
    # train_set, test_set_1 = data_set.split(0.7)

    # 定义模型并训练
    # model = PytorchModel(FcReluFcRelu([128, 64, 1]))
    model = PytorchModel(CNN(2048, 1))
    model.train(train_set, 10, weight_decay=0.01)
    Plotter.loss(model)

    result = model.test(test_set)
    Plotter.rul_end2end(test_set, result)

    # 预测结果评价
    evaluator = Evaluator()
    evaluator.add(End2EndMAE(), End2EndMSE(), End2EndRMSE())
    evaluator(test_set, result)
