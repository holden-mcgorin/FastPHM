import numpy as np
from numpy import ndarray

from fastphm.data.labeler.ABCLabeler import ABCLabeler
from fastphm.data.Dataset import Dataset
from fastphm.data.processor.SlideWindowProcessor import SlideWindowProcessor
from fastphm.entity.Bearing import Bearing


class BearingRulLabeler(ABCLabeler):
    """
    todo 默认轴承的RUL标签都归一化为1
    本框架数据集的x默认形状： (batch, time, features)
    数据的维度建议，维度转换在模型处设置
    RNN / LSTM / GRU / Transformer	(batch, time, features) ✅
    1D CNN	(batch, features, time) ✅
    """

    def __init__(self, window_size: int, window_step=None,
                 time_ratio=1,
                 is_from_fpt=False, is_rectified=False, is_squeeze=True):
        """
        :param window_size:滑动窗口长度
        :param window_step:滑动窗口步长
        :param is_from_fpt:是否fpt后才开始生成数据
        :param is_rectified:fpt之前rul是否固定为1
        :param time_ratio: 生成的数据集的时间单位 / 实体的rul、life等时间单位
        """
        self.window_size = window_size
        self.window_step = window_step if window_step is not None else window_size
        self.time_ratio = time_ratio
        self.is_from_fpt = is_from_fpt
        self.is_rectified = is_rectified
        self.is_squeeze = is_squeeze  # 压缩x中长度为1的轴（只有一个特征时去掉特征轴）
        self.sliding_window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    @property
    def name(self):
        return 'RUL'

    def _label(self, bearing: Bearing) -> Dataset:
        if self.is_from_fpt:
            raw_data = bearing.raw_data.iloc[bearing.stage_data.fpt_raw:, :].values
            x = self.sliding_window(raw_data)
            if self.is_squeeze:
                x = x.squeeze()
            z = np.linspace(0, bearing.rul, x.shape[0]).reshape(-1, 1) / self.time_ratio
        else:
            raw_data = bearing.raw_data.values
            x = self.sliding_window(raw_data)
            if self.is_squeeze:
                x = x.squeeze()
            z = np.linspace(0, bearing.life, x.shape[0]).reshape(-1, 1) / self.time_ratio

        if self.is_rectified and not self.is_from_fpt:
            fpt_index = bearing.stage_data.fpt_raw // self.window_size
            y1 = np.ones((fpt_index, 1))
            y2 = np.linspace(1, 0, x.shape[0] - fpt_index).reshape(-1, 1)
            y = np.vstack((y1, y2))
        else:
            y = np.linspace(1, 0, x.shape[0]).reshape(-1, 1)

        return Dataset(x=x, y=y, z=z, name=bearing.name)
