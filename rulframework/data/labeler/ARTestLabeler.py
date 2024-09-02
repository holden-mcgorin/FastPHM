import numpy as np

from rulframework.data.Dataset import Dataset
from rulframework.data.labeler.ABCLabeler import ABCLabeler
from rulframework.entity.Bearing import Bearing


class ARTestLabeler(ABCLabeler):
    """
    自回归标签生成器
    """

    def __init__(self, begin_index, input_length):
        self.begin_index = begin_index
        self.input_length = input_length

    @property
    def name(self):
        return 'AutoRegressionTest'

    def _label(self, bearing: Bearing) -> Dataset:
        # todo 暂时只支持轴承的第一个特征（未来推广到所有特征，增加可选项对原始信号自回归）
        x = bearing.feature_data.values[self.begin_index - self.input_length:self.begin_index, 0].reshape(1, -1)

        y = bearing.feature_data.values[self.begin_index:bearing.stage_data.eol_feature, 0].reshape(1, -1)

        z = np.array([[self.begin_index]])

        return Dataset(x, y, z, name=bearing.name)
