import numpy as np
from numpy import ndarray

from fastphm.data.processor.ABCProcessor import ABCProcessor


class NormalizationProcessor(ABCProcessor):
    def __init__(self, arr_min=None, arr_max=None):
        self.arr_min = arr_min
        self.arr_max = arr_max

    def __call__(self, source: ndarray) -> ndarray:
        # 对每一列进行归一化处理
        if self.arr_min is None:
            arr_min = np.min(source, axis=0)  # 每列的最小值
        else:
            arr_min = self.arr_min

        if self.arr_max is None:
            arr_max = np.max(source, axis=0)  # 每列的最大值
        else:
            arr_max = self.arr_max

        # 归一化公式 (x - min) / (max - min)
        arr_normalized = (source - arr_min) / (arr_max - arr_min)
        return arr_normalized
