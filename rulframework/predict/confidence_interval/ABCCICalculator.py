from abc import ABC, abstractmethod
from typing import Union


class ABCCICalculator(ABC):
    """
    置信区间计算器
    """

    def __init__(self, arg) -> None:
        """
        :param arg:置信区间参数
        """
        self.arg = arg

    @abstractmethod
    def calculate(self, input_data: list) -> (
            Union[int, float], Union[int, float], Union[int, float]):
        """
        计算每个 x_test 对应置信区间的 y_test 的最小值、均值、最大值
        :param input_data: x_test 对应的 y_test 的采样列表
        :return: min_value, mean_value, max_value
        """
        raise NotImplementedError
