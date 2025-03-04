from typing import List

import numpy as np
from numpy import ndarray

from rulframework.data.labeler.ABCLabeler import ABCLabeler
from rulframework.data.Dataset import Dataset
from rulframework.entity.Bearing import Bearing, Fault


class FaultLabeler(ABCLabeler):
    """
    默认使用类型下标作为标签
    可以设置is_onehot使用独热编码作为标签
    """

    @property
    def name(self):
        return 'fault'

    def __init__(self, interval: int, fault_types: List[Fault], time_ratio=1, is_onehot: bool = True):
        # 特征间隔
        self.interval = interval
        # 所有的故障类型及其次序
        self.fault_types = fault_types
        self.time_ratio = time_ratio
        # 生成的标签是否使用独热编码（当使用pytorch交叉熵时不需要手动计算独热编码）
        self.is_onehot = is_onehot

    def _label(self, bearing: Bearing) -> Dataset:
        # 只取了第一列 todo

        # 生成特征x
        raw_data: ndarray = bearing.raw_data.iloc[:, 0].values
        x = raw_data.reshape(-1, self.interval)

        # 生成标签y
        # 获取健康状态的下标
        try:
            normal_index = self.fault_types.index(Fault.NC)
        except ValueError:
            normal_index = None
        # 获取故障状态的下标
        fault_indices = []
        for i, e in enumerate(self.fault_types):
            if e in bearing.fault_type:
                fault_indices.append(i)
        # 状态数
        cols = len(self.fault_types)

        if self.is_onehot:
            # 生成健康标签
            y_normal = np.zeros((bearing.stage_data.fpt_raw // self.interval, cols))
            if normal_index is not None:
                y_normal[:, normal_index] = 1
            # 生成故障标签
            y_fault = np.zeros(((bearing.raw_data.shape[0] - bearing.stage_data.fpt_raw) // self.interval, cols))
            y_fault[:, fault_indices] = 1
        else:
            # 生成健康标签
            y_normal = np.zeros((bearing.stage_data.fpt_raw // self.interval, 1))
            if normal_index is not None:
                y_normal[:, 0] = normal_index
            # 生成故障标签
            y_fault = np.zeros(((bearing.raw_data.shape[0] - bearing.stage_data.fpt_raw) // self.interval, 1))
            y_fault[:, 0] = fault_indices[0]  # 非独热编码时自适用单标签多分类问题

        # 合并健康阶段与故障阶段
        if y_normal is None:
            y = y_fault
        else:
            y = np.vstack((y_normal, y_fault))

        # 生成时间z
        z = np.linspace(0, bearing.life, x.shape[0]).reshape(-1, 1) / self.time_ratio

        return Dataset(x, y, z, name=bearing.name)
