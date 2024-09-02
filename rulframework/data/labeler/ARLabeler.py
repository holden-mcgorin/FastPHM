import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.data.labeler.ABCLabeler import ABCLabeler
from rulframework.data.processor.SlideWindowProcessor import SlideWindowProcessor
from rulframework.entity.Bearing import Bearing


class ARLabeler(ABCLabeler):
    """
    自回归标签生成器
    """

    def __init__(self, input_length, output_length, window_step=1):
        self.window_size_x = input_length
        self.window_size_y = output_length
        self.window_step = window_step

    @property
    def name(self):
        return 'AutoRegression'

    def _label(self, bearing: Bearing) -> Dataset:
        slide_window = SlideWindowProcessor(self.window_size_x + self.window_size_y, self.window_step)

        # todo 暂时只支持轴承的第一个特征（未来推广到所有特征，增加可选项对原始信号自回归）
        # 滑动窗口的前半部分x为特征，后半部分y为标签
        data = bearing.feature_data.values
        xy = slide_window.__call__(data[:, 0].reshape(-1))
        x = xy[:, :self.window_size_x]
        y = xy[:, self.window_size_x:]

        z = np.linspace(0, bearing.life, x.shape[0]).reshape(-1, 1)

        return Dataset(x, y, z, name=bearing.name)
