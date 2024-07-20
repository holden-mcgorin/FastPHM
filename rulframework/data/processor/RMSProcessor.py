from rulframework.data.processor.ABCProcessor import ABCProcessor

import numpy as np
from numpy import ndarray


class RMSProcessor(ABCProcessor):
    def __init__(self, window_size: int, window_step: int = -1):
        """
        :param window_size: 用于计算RMS的区间大小
        :param window_step: 窗口的步长，默认为-1，如果为-1则步长等于窗口大小
        """
        self.window_size = window_size
        self.window_step = window_step if window_step != -1 else window_size

    def __call__(self, source: ndarray) -> ndarray:
        # 计算滑动窗口的数量
        num_windows = (len(source) - self.window_size) // self.window_step + 1

        # 初始化存储RMS结果的数组
        target = np.zeros(num_windows)

        for i in range(num_windows):
            start_idx = i * self.window_step
            end_idx = start_idx + self.window_size
            window = source[start_idx:end_idx]
            target[i] = np.sqrt(np.mean(window ** 2))

        return target
