import numpy as np

from fastphm.data.Dataset import Dataset
from fastphm.model.Result import Result
from fastphm.metric.ABCMetric import ABCMetric


class MAE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MAE'

    @property
    def is_higher_better(self) -> bool:
        return False

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.y_hat
        r = test_set.y
        return float(np.mean(np.abs(r - r_hat)))
