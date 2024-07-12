import numpy as np
from numpy import ndarray

from rulframework.data.dataset.ABCGenerator import ABCGenerator
from rulframework.data.dataset.Dataset import Dataset
from rulframework.entity.Bearing import Bearing


class RelativeRULGenerator(ABCGenerator):
    def __init__(self, interval, is_from_fpt=True, is_rectified=False):
        """
        :param interval:每个样本的振动信号区间长度
        :param is_from_fpt:是否fpt后才开始生成数据
        :param is_rectified:fpt之前rul是否固定为1
        """
        self.interval = interval
        self.is_from_fpt = is_from_fpt
        self.is_rectified = is_rectified

    @property
    def name(self):
        return 'rul'

    def _generate(self, bearing: Bearing) -> Dataset:
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
            z = np.linspace(0, bearing.rul, x.shape[0]).reshape(-1, 1)
        else:
            z = np.linspace(0, bearing.life, x.shape[0]).reshape(-1, 1)

        return Dataset(x, y, z, name=bearing.name)
