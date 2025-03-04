import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric


class MAE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MAE'

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.outputs
        r = test_set.y
        return float(np.mean(np.abs(r - r_hat)))
