import numpy as np

from rulframework.data.dataset.Dataset import Dataset
from rulframework.predict.Result import Result
from rulframework.predict.evaluator.end2end_metric.ABCEnd2EndMetric import ABCEnd2EndMetric


class End2EndMSE(ABCEnd2EndMetric):
    @property
    def name(self) -> str:
        return 'MSE'

    def measure(self, test_set: Dataset, result: Result):
        sample_num = test_set.x.shape[0]
        if sample_num != result.mean.shape[0]:
            raise Exception(f'测试样本量：{sample_num}与测试结果数量：{result.mean.shape[0]} 不匹配')
        r_hat = result.mean
        r = test_set.y
        return f"{float(np.mean((r_hat - r) ** 2, axis=0)):.4f}"
