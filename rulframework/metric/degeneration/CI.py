import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.metric.ABCMetric import ABCMetric
from rulframework.model.Result import Result


class CI(ABCMetric):
    @property
    def name(self) -> str:
        return '95%CI'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        begin, end = len(result.upper), len(result.lower)
        for i in range(len(result.upper)):
            if result.upper[i] > test_set.y[0][-1]:
                begin = i
                break
        for i in range(len(result.upper)):
            if result.lower[i] > test_set.y[0][-1]:
                end = i
                break
        return f'[{begin}, {end}]'
