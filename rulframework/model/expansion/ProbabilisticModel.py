import sys
from typing import Union

import numpy as np
from numpy import ndarray

from rulframework.data.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel
from rulframework.model.Result import Result
from rulframework.model.expansion.ABCExpansionModel import ABCExpansionModel
from rulframework.model.expansion.ci.ABCConfidenceInterval import ABCConfidenceInterval


class ProbabilisticModel(ABCExpansionModel):
    """
    输出形状：(批量大小, 采样数, 预测结果)
    """

    def __init__(self, model: ABCModel, ci: ABCConfidenceInterval, samples: int) -> None:
        super().__init__(model)
        self.ci = ci
        self.samples = samples

    def __call__(self, x: ndarray) -> Union[tuple, ndarray]:
        # 生成多个采样结果
        all_samples = []
        for i in range(self.samples):
            all_samples.append(self.model(x))

        # 输出形状：(批量大小, 采样数, 预测结果)
        return np.stack(all_samples, axis=1)

    def test(self, test_set: Dataset) -> Result:
        result = Result(begin_index=test_set.z.squeeze(), outputs=self(test_set.x))
        result = self.ci(result)
        return result
