import torch
from torch import nn

from rulframework.data.dataset.Dataset import Dataset
from rulframework.data.dataset.FaultLabelGenerator import FaultLabelGenerator
from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.dataset.RelativeRULGenerator import RelativeRULGenerator
from rulframework.entity.Bearing import Bearing
from rulframework.model.PytorchModel import PytorchModel
from rulframework.model.mlp.FcReluFcRelu import FcReluFcRelu
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.model.mlp.FcReluFcSoftmax import FcReluFcSoftmax
from rulframework.predict.evaluator.End2EndEvaluator import End2EndEvaluator
from rulframework.predict.evaluator.end2end_metric.End2EndMSE import End2EndMSE
from rulframework.predict.evaluator.end2end_metric.End2EndRMSE import End2EndRMSE
from rulframework.util.Plotter import Plotter
import torch.nn.functional as F

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
    generator = FaultLabelGenerator(128, list(Bearing.FaultType.__members__.values()))
    dataset = generator.generate(bearing)
    # train_set, test_set = dataset.split(0.7)
    # train_set = Dataset()

    # 通过其他轴承增加训练数据
    for bearing_name in ['Bearing1_1', 'Bearing1_4', 'Bearing2_1', 'Bearing1_2']:
        print(f'正在使用{bearing_name}构造训练数据')
        bearing_train = data_loader.get_bearing(bearing_name)
        bearing_train.feature_data = feature_extractor.extract(bearing_train.raw_data)
        stage_calculator.calculate_state(bearing_train)
        dataset.append(generator.generate(bearing_train))

    train_set, test_set = dataset.split(0.7)

    # 定义模型并训练
    model = PytorchModel(FcReluFcSoftmax([128, 64, 5]), criterion=nn.MSELoss())
    # model = PytorchModel(FcReluFcSoftmax([128, 64, 5]), criterion=nn.CrossEntropyLoss())

    model.end2end_train(train_set, 10, weight_decay=0.01)
    Plotter.loss(model)

    result = model.end2end_predict(test_set)
    # result.mean = F.softmax(torch.from_numpy(result.mean), dim=1).numpy()

    # Plotter.fault_during_time(test_set, result, interval=1)
    Plotter.fault_prediction_heatmap(test_set, result)

    #
    # # 预测结果评价
    # evaluator = End2EndEvaluator()
    # evaluator.add_metric(End2EndRMSE(), End2EndMSE())
    # evaluator.evaluate(test_set, result)
