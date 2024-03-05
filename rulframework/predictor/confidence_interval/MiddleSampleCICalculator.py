from typing import Union

from rulframework.predictor.confidence_interval.ABCCICalculator import ABCCICalculator


class MiddleSampleCICalculator(ABCCICalculator):

    def __init__(self, arg) -> None:
        """
        :param arg: 区间率，取值0~1
        """
        super().__init__(arg)

    def calculate(self, input_data: list) -> (
            Union[int, float], Union[int, float], Union[int, float]):
        """
        取采样的中间样本做置信区间
        :param input_data: 采样得到的数据
        :return: min_value, mean_value, max_value
        """
        sampling_num = len(input_data)
        lower_index = int(sampling_num * ((1 - self.arg) // 2))  # 下边界索引
        upper_index = int(sampling_num * self.arg + ((1 - self.arg) // 2))  # 上边界索引

        sorted_list = sorted(input_data)
        new_list = sorted_list[lower_index:upper_index]
        # 取区间内的最大值、最小值、平均值
        max_value = max(new_list)
        min_value = min(new_list)
        mean_value = sum(new_list) / len(new_list)
        return min_value, mean_value, max_value
