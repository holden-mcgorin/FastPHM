import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.metric.ABCMetric import ABCMetric
from rulframework.model.Result import Result


class PercentError(ABCMetric):
    @property
    def name(self) -> str:
        return 'Percent Error'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        r_hat = result.outputs.reshape(-1)
        r = test_set.y.reshape(-1)

        # 去掉数值为0的元素
        zero_indices = np.where(r == 0)[0]
        r_hat = np.delete(r_hat, zero_indices)
        r = np.delete(r, zero_indices)

        return f"{float(np.mean((r - r_hat) / r, axis=0)) * 100:.2f}%"
