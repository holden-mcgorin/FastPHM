from typing import Union
import statistics

from rulframework.predict.confidence_interval.ABCCICalculator import ABCCICalculator


class MeanPlusStdCICalculator(ABCCICalculator):

    def __init__(self, arg) -> None:
        """
        :param arg: 均值 + arg * 标准差
        """
        super().__init__(arg)

    def calculate(self, input_data: list) -> (
            Union[int, float], Union[int, float], Union[int, float]):
        """
        取均值和标准差来计算置信区间
        :param input_data: 某个x对应的y的采样列表
        :return: min_value, mean_value, max_value
        """
        sampling_num = len(input_data)
        lower_index = int(sampling_num * ((1 - 0.9) // 2))  # 下边界索引
        upper_index = int(sampling_num * 0.9 + ((1 - 0.9) // 2))  # 上边界索引

        sorted_list = sorted(input_data)
        new_list = sorted_list[lower_index:upper_index]

        mean_value = statistics.mean(new_list)
        std = statistics.stdev(new_list)
        min_value = mean_value - self.arg * std
        max_value = mean_value + self.arg * std
        return min_value, mean_value, max_value
