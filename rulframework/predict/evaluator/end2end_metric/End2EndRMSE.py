import numpy as np

from rulframework.data.dataset.Dataset import Dataset
from rulframework.predict.Result import Result
from rulframework.predict.evaluator.end2end_metric.ABCEnd2EndMetric import ABCEnd2EndMetric


class End2EndRMSE(ABCEnd2EndMetric):
    @property
    def name(self) -> str:
        return 'RMSE'

    def _measure(self, test_set: Dataset, result: Result):
        r_hat = result.mean
        r = test_set.y
        return f"{float(np.sqrt(np.mean((r_hat - r) ** 2, axis=0))):.4f}"
