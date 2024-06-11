import numpy as np
from numpy import ndarray

from rulframework.data.dataset.ABCGenerator import ABCGenerator
from rulframework.data.dataset.Dataset import Dataset
from rulframework.entity.Bearing import Bearing


class RelativeRULGenerator(ABCGenerator):
    def __init__(self, interval):
        self.interval = interval

    def generate(self, bearing: Bearing) -> Dataset:
        # 只取了第一列 todo
        raw_data: ndarray = bearing.raw_data.iloc[bearing.stage_data.fpt_raw:, 0].values
        x = raw_data.reshape(-1, self.interval)
        y = np.linspace(1, 0, x.shape[0]).reshape(-1, 1)
        return Dataset(x, y, name=bearing.name)
