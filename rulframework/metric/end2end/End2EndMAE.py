import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric


class End2EndMAE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MAE'

    def __call__(self, test_set: Dataset, result: Result):
        r_hat = result.outputs
        r = test_set.y
        mae = np.mean(np.abs(r - r_hat))
        return f"{mae:.4f}"
