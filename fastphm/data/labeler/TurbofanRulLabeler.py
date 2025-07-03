import numpy as np
from numpy import ndarray

from fastphm.data.labeler.ABCLabeler import ABCLabeler
from fastphm.data.Dataset import Dataset
from fastphm.data.processor.SlideWindowProcessor import SlideWindowProcessor
from fastphm.entity.ABCEntity import ABCEntity
from fastphm.entity.Bearing import Bearing
from fastphm.entity.Turbofan import Turbofan


class TurbofanRulLabeler(ABCLabeler):

    def __init__(self, window_size, window_step=1, max_rul=-1, last_sample: bool = False):
        """
        涡扇发动机数据打标器
        对于涡扇发动机数据集来说，评价指标通常只评价整个发动机的RUL，而不是滑动窗口生成的所有切片，
        对于常使用的score指标来说，测试集越多分数会越大，因此在某些情况下需要仅取每个发动机最后一个样本作为测试集评价预测结果
        :param window_size:
        :param window_step:
        :param max_rul:
        :param last_sample: 是否只取每个发动机最后一个样本（该选择对score评价指标至关重要）
        """
        self.window_size = window_size  # 为-1代表窗口大小为整个时序长度
        self.window_step = window_step
        self.max_rul = max_rul
        self.last_sample = last_sample
        self.sliding_window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    @property
    def name(self):
        return 'RUL'

    def _label(self, turbofan: Turbofan) -> Dataset:
        data = turbofan.feature_data.values
        x = self.sliding_window(data)
        y = np.arange(x.shape[0], 0, -1).reshape(-1, 1) + turbofan.rul - 1
        y[y > self.max_rul] = self.max_rul
        z = np.arange(x.shape[0]) + 1
        z = z.reshape(-1, 1)

        # 是否仅取最后一个样本
        if self.last_sample:
            x = x[np.newaxis, -1]
            y = y[np.newaxis, -1]
            z = z[np.newaxis, -1]

        return Dataset(x=x, y=y, z=z, name=turbofan.name)
