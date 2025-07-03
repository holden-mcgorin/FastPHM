from typing import List

import numpy as np
from numpy import ndarray

from fastphm.data.labeler.ABCLabeler import ABCLabeler
from fastphm.data.Dataset import Dataset
from fastphm.data.processor.SlideWindowProcessor import SlideWindowProcessor
from fastphm.entity.Bearing import Bearing, Fault


class BearingFaultLabeler(ABCLabeler):
    """
    默认使用类型下标作为标签
    可以设置is_onehot使用独热编码作为标签
    """

    @property
    def name(self):
        return 'fault'

    def __init__(self, window_size: int, fault_types: List[Fault], window_step=None,
                 time_ratio=1,
                 is_multi_hot=True, is_from_fpt=False, is_squeeze=True):
        """

        :param window_size:
        :param fault_types:
        :param window_step:
        :param time_ratio: 生成的数据集的时间单位 / 实体的rul、life等时间单位
        :param is_multi_hot: 多热编码如(0,1,1)、(1,0,0)否则为类别编码如(2)、(0)
        :param is_from_fpt:
        :param is_squeeze:
        """

        # 滑动窗口大小
        self.window_size = window_size
        # 滑动窗口步长
        self.window_step = window_step if window_step is not None else window_size

        # 所有的故障类型及其次序
        self.fault_types = fault_types
        self.time_ratio = time_ratio
        # 生成的标签是否使用独热编码（当使用pytorch交叉熵时不需要手动计算独热编码）
        self.is_multi_hot = is_multi_hot
        self.is_from_fpt = is_from_fpt
        self.is_squeeze = is_squeeze
        self.sliding_window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    def _label(self, bearing: Bearing) -> Dataset:
        # 仅取退化阶段数据
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

        if self.is_multi_hot:
            # 生成健康标签
            y_normal = np.zeros((bearing.stage_data.fpt_raw // self.window_size, cols))
            if normal_index is not None:
                y_normal[:, normal_index] = 1
            # 生成故障标签
            y_fault = np.zeros(((bearing.raw_data.shape[0] - bearing.stage_data.fpt_raw) // self.window_size, cols))
            y_fault[:, fault_indices] = 1
        else:
            # 生成健康标签
            y_normal = np.zeros((bearing.stage_data.fpt_raw // self.window_size, 1))
            if normal_index is not None:
                y_normal[:, 0] = normal_index
            # 生成故障标签
            y_fault = np.zeros(((bearing.raw_data.shape[0] - bearing.stage_data.fpt_raw) // self.window_size, 1))
            y_fault[:, 0] = fault_indices[0]  # 非独热编码时自适用单标签多分类问题

        # 合并健康阶段与故障阶段
        if y_normal is None or self.is_from_fpt:
            y = y_fault
        else:
            y = np.vstack((y_normal, y_fault))

        return Dataset(x=x, y=y, z=z, name=bearing.name)
