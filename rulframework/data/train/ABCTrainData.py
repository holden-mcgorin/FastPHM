from abc import ABC, abstractmethod

from numpy import ndarray
from pandas import DataFrame

from rulframework.entity.Bearing import Bearing


class ABCTrainData(ABC):
    """
    所有数据生成器的抽象基类
    """

    @abstractmethod
    def generate_data(self, bearing: Bearing) -> (ndarray, ndarray):
        """
        @param bearing: 源数据，一般为特征数据feature_data
        @return generated_data: 返回生成的数据
        """
        pass
