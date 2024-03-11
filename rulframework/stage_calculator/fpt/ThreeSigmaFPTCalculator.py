import numpy as np
from pandas import DataFrame

from rulframework.stage_calculator.fpt.ABCFPTCalculator import ABCFPTCalculator


class ThreeSigmaFPTCalculator(ABCFPTCalculator):
    def get_fpt(self, raw_data: DataFrame, feature_data: DataFrame, scale: int) -> (int, int):
        fpt_feature = 0
        feature_data = feature_data.iloc[:, 0]  # todo 这里只取第一列做计算fpt，多列情况不适应
        for i in range(1, len(feature_data)):
            sliced_list = feature_data[:i + 1]
            if max(sliced_list) > self.__mean_plus_3std(sliced_list):
                fpt_feature = i
                break
        fpt_raw = scale * fpt_feature
        return fpt_raw, fpt_feature

    @staticmethod
    def __mean_plus_3std(signal) -> int:
        mean_value = np.mean(signal)
        std_dev = np.std(signal)
        result = mean_value + 3 * std_dev
        return result
