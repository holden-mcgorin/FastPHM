from abc import ABC, abstractmethod
from pandas import DataFrame


class ABCFeatureExtractor(ABC):
    """
    所有特征提取器的抽象基类
    """
    @abstractmethod
    def extract(self, raw_data) -> DataFrame:
        """
        从原始数据中提取特征
        :return:feature
        """
        pass
