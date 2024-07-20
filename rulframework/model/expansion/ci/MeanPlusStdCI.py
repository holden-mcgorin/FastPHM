import numpy as np

from rulframework.model.Result import Result
from rulframework.model.expansion.ci.ABCConfidenceInterval import ABCConfidenceInterval


class MeanPlusStdCI(ABCConfidenceInterval):
    """
    计算平均数和标准差，使用平均数±n倍标准差作为置信区间
    """

    def __init__(self, rate: float) -> None:
        """
        计算均值±n标准差
        :param rate:标准差的倍率
        """
        self.rate = rate

    def __call__(self, result: Result) -> Result:
        arr = result.outputs
        mean = np.mean(arr, axis=1).reshape(-1)
        std = np.std(arr, axis=1).reshape(-1)

        result.mean = mean
        result.lower = mean - self.rate * std
        result.upper = mean + self.rate * std

        return result
