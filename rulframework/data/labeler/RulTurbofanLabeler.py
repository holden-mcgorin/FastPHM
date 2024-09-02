import numpy as np
from numpy import ndarray

from rulframework.data.labeler.ABCLabeler import ABCLabeler
from rulframework.data.Dataset import Dataset
from rulframework.data.processor.SlideWindowProcessor import SlideWindowProcessor
from rulframework.entity.ABCEntity import ABCEntity
from rulframework.entity.Bearing import Bearing


class RulTurbofanLabeler(ABCLabeler):
    """

    """

    def __init__(self, window_size, window_step=1, max_rul=-1):
        """
        涡扇发动机数据打标器
        :param window_size:
        :param window_step:
        :param max_rul:
        """
        self.window_size = window_size
        self.window_step = window_step
        self.max_rul = max_rul
        self.window = SlideWindowProcessor(window_size=self.window_size, window_step=self.window_step)

    @property
    def name(self):
        return 'RUL'

    def _label(self, entity: ABCEntity) -> Dataset:
        data = entity.feature_data.values
        x = self.window(data)
        y = np.arange(x.shape[0], 0, -1).reshape(-1, 1)
        y[y > self.max_rul] = self.max_rul
        z = np.arange(x.shape[0]) + 1
        z = z.reshape(-1, 1)

        return Dataset(x, y, z, name=entity.name)
