import numpy as np
from numpy import ndarray

from fastphm.data.labeler.ABCLabeler import ABCLabeler
from fastphm.data.Dataset import Dataset
from fastphm.entity.Bearing import Bearing


class RulLabeler(ABCLabeler):
    """

    """

    def __init__(self, interval, interval_step=-1, time_ratio=1,
                 is_from_fpt=True, is_rectified=False, is_relative=True):
        """
        :param interval:每个样本的振动信号区间长度
        :param interval_step:interval的步长
        :param is_from_fpt:是否fpt后才开始生成数据
        :param is_rectified:fpt之前rul是否固定为1
        :param is_relative:是否是归一化rul
        """
        self.interval = interval
        self.interval_step = interval  # todo
        self.time_ratio = time_ratio
        self.is_from_fpt = is_from_fpt
        self.is_rectified = is_rectified
        self.is_relative = is_relative  # 待完成 todo

    @property
    def name(self):
        return 'RUL'

    def _label(self, bearing: Bearing) -> Dataset:
        # 只取了第一列 todo
        if self.is_from_fpt:
            raw_data: ndarray = bearing.raw_data.iloc[bearing.stage_data.fpt_raw:, 0].values
        else:
            raw_data = bearing.raw_data.values

        x = raw_data.reshape(-1, self.interval)

        if self.is_rectified:
            fpt_index = bearing.stage_data.fpt_raw // self.interval
            y1 = np.ones((fpt_index, 1))
            y2 = np.linspace(1, 0, x.shape[0] - fpt_index).reshape(-1, 1)
            y = np.vstack((y1, y2))
        else:
            y = np.linspace(1, 0, x.shape[0]).reshape(-1, 1)

        if self.is_from_fpt:
            z = np.linspace(0, bearing.rul, x.shape[0]).reshape(-1, 1) / self.time_ratio
        else:
            z = np.linspace(0, bearing.life, x.shape[0]).reshape(-1, 1) / self.time_ratio

        return Dataset(x, y, z, name=bearing.name)
