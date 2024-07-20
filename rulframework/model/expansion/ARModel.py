import sys
from typing import Union

import numpy as np
from numpy import ndarray

from rulframework.data.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel
from rulframework.model.Result import Result
from rulframework.model.expansion.ABCExpansionModel import ABCExpansionModel
from rulframework.system.Logger import Logger


class ARModel(ABCExpansionModel):
    """
    在模型基础上的扩展预测器
    增加额外的功能
    这个是自回归扩展预测器
    """

    def __init__(self, model: ABCModel, epochs, threshold: float = sys.float_info.max) -> None:
        super().__init__(model)
        self.epochs = epochs
        if threshold != sys.float_info.max:
            Logger.warning('如果使用概率模型对回归模型套娃的话建议不要设置阈值，可能会导致概率模型每次采样的形状不一样，建议直接在画图时设置裁剪!')
        self.threshold = threshold

    def __call__(self, x: ndarray) -> Union[tuple, ndarray]:

        length_x = x.shape[-1]
        history = self.model(x)
        if x.ndim != history.ndim:
            x = np.expand_dims(x, axis=1)
            x = np.repeat(x, repeats=history.shape[1], axis=1)
        for i in range(1, self.epochs):
            x = np.concatenate([x, history], axis=-1)[..., -length_x:]
            y_hat = self.model(x)
            history = np.concatenate([history, y_hat], axis=-1)
            if np.any(x > self.threshold):
                break

        return history

    def test(self, test_set: Dataset) -> Result:
        return Result(begin_index=test_set.z.squeeze(), outputs=self(test_set.x))
