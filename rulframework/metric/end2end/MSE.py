import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric


class MSE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MSE'

    def __call__(self, test_set: Dataset, result: Result):
        r_hat = result.outputs
        r = test_set.y
        return f"{float(np.mean((r_hat - r) ** 2, axis=0)):.4f}"
