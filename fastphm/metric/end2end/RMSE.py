import numpy as np

from fastphm.data.Dataset import Dataset
from fastphm.model.Result import Result
from fastphm.metric.ABCMetric import ABCMetric


class RMSE(ABCMetric):
    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.outputs
        r = test_set.y
        return float(np.sqrt(np.mean((r_hat - r) ** 2, axis=0)))

    @property
    def name(self) -> str:
        return 'RMSE'
