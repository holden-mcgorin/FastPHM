from __future__ import annotations

import numpy as np
from numpy import ndarray


class Result:
    """
    轴承预测数据
    """

    # todo 需要重新设置数据结构，使得可以支持分开保存、拼接多个模型的预测结果（已实现内部拼接，还需实现外部拼接）

    def __init__(self, name: str = 'prediction', begin_index: int = None, outputs: ndarray = None,
                 upper: ndarray = None, mean: ndarray = None, lower: ndarray = None) -> None:
        """
        :param begin_index: 开始预测时的下标
        :param outputs: 模型的所有输出，形状：(批量大小,预测结果)或(批量大小, 采样数, 预测结果)
        :param upper: 不确定性区间上界，形状：(预测结果)
        :param mean: 不确定性预测均值，形状：(预测结果)
        :param lower: 不确定性区间下界，形状：(预测结果)
        """
        self.name = name
        self.begin_index = begin_index
        self.outputs = outputs
        self.upper = upper
        self.mean = mean
        self.lower = lower

    def __copy__(self):
        return Result(
            name=self.name if self.name is not None else None,
            begin_index=self.begin_index if self.begin_index is not None else None,
            outputs=self.outputs.copy() if self.outputs is not None else None,
            upper=self.upper.copy() if self.upper is not None else None,
            mean=self.mean.copy() if self.mean is not None else None,
            lower=self.lower.copy() if self.lower is not None else None
        )

    def __str__(self) -> str:
        return f"begin_index = {self.begin_index}\noutputs = {self.outputs}"

    def append(self, another_result: Result):
        # todo 待完善
        if self.outputs is None:
            self.outputs = another_result.outputs
        else:
            self.outputs = np.vstack((self.outputs, another_result.outputs))

    @property
    def is_empty(self):
        if self.outputs is None and self.mean is None and self.lower is None and self.upper is None:
            return True
        else:
            return False
