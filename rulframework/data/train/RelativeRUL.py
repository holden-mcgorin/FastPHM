import numpy as np
from numpy import ndarray

from rulframework.entity.Bearing import Bearing


class RelativeRUL:
    def generate(self, bearing: Bearing, interval: int) -> (ndarray, ndarray):
        # 只取了第一列 todo
        raw_data: ndarray = bearing.raw_data.iloc[bearing.stage_data.fpt_raw:, 0].values
        x = raw_data.reshape(-1, interval)
        y = np.linspace(1, 0, x.shape[0]).reshape(-1, 1)
        return x, y
