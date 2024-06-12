from typing import List

import numpy as np
from numpy import ndarray

from rulframework.data.dataset.ABCGenerator import ABCGenerator
from rulframework.data.dataset.Dataset import Dataset
from rulframework.entity.Bearing import Bearing


class FaultLabelGenerator(ABCGenerator):
    def __init__(self, interval: int, fault_types: List[Bearing.FaultType]):
        self.interval = interval
        self.fault_types = fault_types

    def generate(self, bearing: Bearing) -> Dataset:
        # 只取了第一列 todo
        raw_data: ndarray = bearing.raw_data.iloc[:, 0].values
        normal_index = self.fault_types.index(Bearing.FaultType.NC)
        fault_indices = []
        for i, e in enumerate(self.fault_types):
            if e in bearing.fault_type:
                fault_indices.append(i)

        x = raw_data.reshape(-1, self.interval)
        cols = len(self.fault_types)
        y_normal = np.zeros((bearing.stage_data.fpt_raw // self.interval, cols))
        y_normal[:, normal_index] = 1
        y_fault = np.zeros(((bearing.raw_data.shape[0] - bearing.stage_data.fpt_raw) // self.interval, cols))
        y_fault[:, fault_indices] = 1
        y = np.vstack((y_normal, y_fault))

        z = np.linspace(0, bearing.total_life, x.shape[0]).reshape(-1, 1)

        return Dataset(x, y, z, name=bearing.name)
