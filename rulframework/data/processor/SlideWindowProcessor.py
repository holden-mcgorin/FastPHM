import numpy as np
from numpy import ndarray

from rulframework.data.processor.ABCProcessor import ABCProcessor


class SlideWindowProcessor(ABCProcessor):
    def __init__(self, window_size: int, window_step: int = 1):
        """
        :param window_size:用于计算RMS的区间大小
        """
        self.window_size = window_size
        self.window_step = window_step

    def __call__(self, source: ndarray) -> ndarray:
        # 确定窗口的数量
        num_windows = (len(source) - self.window_size) // self.window_step + 1
        # 初始化存储窗口数据的数组
        windows = np.zeros((num_windows, self.window_size))

        for i in range(num_windows):
            start_index = i * self.window_step
            end_index = start_index + self.window_size
            windows[i] = source[start_index:end_index]

        return windows
