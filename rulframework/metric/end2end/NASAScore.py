import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.metric.ABCMetric import ABCMetric
from rulframework.model.Result import Result


class NASAScore(ABCMetric):
    @property
    def name(self) -> str:
        return 'NASA Score'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        r_hat = result.outputs.reshape(-1)
        r = test_set.y.reshape(-1)

        # 初始化 Score
        score = 0.0

        # 计算每个 S_i 并累加
        for i in range(len(r_hat)):
            if r_hat[i] >= r[i]:
                s = np.exp((r_hat[i] - r[i]) / 10) - 1
            else:
                s = np.exp((r[i] - r_hat[i]) / 13) - 1

            score += s

        return f"{float(score):.4f}"
