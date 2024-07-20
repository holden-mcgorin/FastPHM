import numpy as np

from rulframework.model.Result import Result
from rulframework.model.expansion.ci.ABCConfidenceInterval import ABCConfidenceInterval


class MiddleSampleCI(ABCConfidenceInterval):
    """
    将采样点排序，取中间的数据作为置信区间
    """

    def __init__(self, rate: float) -> None:
        """
        :param rate:区间率，取值0~1，指占原始数据的比率
        """
        self.rate = rate

    def __call__(self, result: Result) -> Result:
        arr = result.outputs

        result.mean = np.mean(arr, axis=1).reshape(-1)
        result.lower = np.quantile(arr, 0.5 - self.rate / 2, axis=0)
        result.upper = np.quantile(arr, 0.5 + self.rate / 2, axis=0)
        return result
