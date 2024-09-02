import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.metric.ABCMetric import ABCMetric
from rulframework.model.Result import Result


class PHM2012Score(ABCMetric):
    @property
    def name(self) -> str:
        return 'PHM2012 Score'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        r_hat = result.outputs
        r = test_set.y

        # 重新排列矩阵的行
        sorted_indices = np.argsort(r, axis=0)
        r = r[sorted_indices]
        r_hat = r_hat[sorted_indices]

        split_r_hat = np.array_split(r_hat, 11, axis=0)
        split_r = np.array_split(r, 11, axis=0)
        errors = np.empty([11], dtype=float)
        scores = np.empty([11], dtype=float)

        for i in range(11):
            # 去掉数值为0的元素
            zero_indices = np.where(split_r[i] == 0)[0]
            split_r_hat[i] = np.delete(split_r_hat[i], zero_indices)
            split_r[i] = np.delete(split_r[i], zero_indices)

            errors[i] = float(np.mean((split_r[i] - split_r_hat[i]) / split_r[i], axis=0)) * 100
            scores[i] = self.score(errors[i])
        return f"{float(np.mean(scores)):.4f}"

    @staticmethod
    def score(percent_error):
        if percent_error <= 0:
            score = np.exp(-np.log(0.5) * (percent_error / 5))
        else:
            score = np.exp(np.log(0.5) * (percent_error / 20))
        return score
