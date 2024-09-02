import numpy as np
from numpy import ndarray

from rulframework.data.processor.ABCProcessor import ABCProcessor


class SlideWindowProcessor(ABCProcessor):
    """
    使用滑动窗口处理后会多一个维度在最外部（轴0）
    """

    def __init__(self, window_size: int, window_step: int = 1, axis: int = 0):
        """
        :param window_size: 用于滑动窗口的区间大小
        :param window_step: 滑动窗口的步长，默认为1
        :param axis: 沿哪个轴进行滑动窗口操作，默认为第0轴
        """
        self.window_size = window_size
        self.window_step = window_step
        self.axis = axis

    def __call__(self, source: ndarray) -> ndarray:
        axis = self.axis

        # 将指定轴移到最前面，以方便滑动窗口操作
        source = np.moveaxis(source, axis, 0)

        # 计算滑动窗口的数量
        num_windows = (source.shape[0] - self.window_size) // self.window_step + 1

        # 初始化存储窗口数据的数组
        new_shape = (num_windows, self.window_size) + source.shape[1:]
        windows = np.zeros(new_shape)

        for i in range(num_windows):
            start_index = i * self.window_step
            end_index = start_index + self.window_size
            windows[i] = source[start_index:end_index]

        # 将轴移回原来的位置
        return np.moveaxis(windows, 0, axis)
