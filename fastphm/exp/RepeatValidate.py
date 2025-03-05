from fastphm.entity.Bearing import Bearing
from fastphm.exp.ABCExperiment import ABCExperiment
from fastphm.system.Logger import Logger
from fastphm.util.Cache import Cache


class RepeatValidate:
    def __init__(self, exp: ABCExperiment, is_save_best_model: bool = False):
        self.exp = exp
        self.is_save_best_model = is_save_best_model

    def repeat(self, bearings: [Bearing], base_bearing, epochs: int):
        train_set, test_set = self.exp.get_dataset(bearings, base_bearing)
        evaluations = []
        best_model = None
        best_metric = None

        for i in range(epochs):
            Logger.info(f'开始第{i + 1}次重复实验：')
            model = self.exp.get_model(train_set)
            e = self.exp.test(model, test_set)
            evaluations.append(e)

            if best_metric is None or e > best_metric:
                best_metric = e
                best_model = model
                Logger.info(f'最佳结果更新：{best_metric}')

            Logger.info(f'重复实验结束，最佳结果：{best_metric}')

            if self.is_save_best_model:
                Cache.save(best_model, 'best_model')

        return evaluations
