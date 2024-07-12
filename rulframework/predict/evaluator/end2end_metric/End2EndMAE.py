import numpy as np

from rulframework.data.dataset.Dataset import Dataset
from rulframework.predict.Result import Result
from rulframework.predict.evaluator.end2end_metric.ABCEnd2EndMetric import ABCEnd2EndMetric


class End2EndMAE(ABCEnd2EndMetric):
    @property
    def name(self) -> str:
        return 'MAE'

    def _measure(self, test_set: Dataset, result: Result):
        r_hat = result.mean
        r = test_set.y
        mae = np.mean(np.abs(r - r_hat))
        return f"{mae:.4f}"
