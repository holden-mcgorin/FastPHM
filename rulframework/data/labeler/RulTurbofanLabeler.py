import numpy as np
from numpy import ndarray

from rulframework.data.labeler.ABCLabeler import ABCLabeler
from rulframework.data.Dataset import Dataset
from rulframework.data.processor.SlideWindowProcessor import SlideWindowProcessor
from rulframework.entity.ABCEntity import ABCEntity
from rulframework.entity.Bearing import Bearing
from rulframework.entity.Turbofan import Turbofan


class RulTurbofanLabeler(ABCLabeler):
    """

    """

    def __init__(self, window_size, window_step=1, max_rul=-1, last_cycle: bool = False):
        """
        涡扇发动机数据打标器
        对于涡扇发动机数据集来说，评价指标通常只评价整个发动机的RUL，而不是滑动窗口生成的所有切片，
        对于常使用的score指标来说，测试集越多分数会越大，因此在某些情况下需要仅取每个发动机最后一个样本作为测试集评价预测结果
        :param window_size:
        :param window_step:
        :param max_rul:
        :param last_cycle: 是否只取每个发动机最后一个样本（该选择对score评价指标至关重要）
        """
        self.window_size = window_size
        self.window_step = window_step
        self.max_rul = max_rul
        self.last_cycle = last_cycle
        self.window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    @property
    def name(self):
        return 'RUL'

    def _label(self, turbofan: Turbofan) -> Dataset:
        data = turbofan.feature_data.values
        x = self.window(data)
        x = x.transpose((0, 2, 1))  # (批量,时间,传感器) -> (批量,传感器,时间)
        y = np.arange(x.shape[0], 0, -1).reshape(-1, 1) + turbofan.rul - 1
        y[y > self.max_rul] = self.max_rul
        z = np.arange(x.shape[0]) + 1
        z = z.reshape(-1, 1)

        # 是否仅取最后一个样本
        if self.last_cycle:
            x = x[np.newaxis, -1]
            y = y[np.newaxis, -1]
            z = z[np.newaxis, -1]

        return Dataset(x, y, z, name=turbofan.name)
