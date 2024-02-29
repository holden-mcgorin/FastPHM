from abc import ABC, abstractmethod

from pandas import DataFrame


class ABCDataGenerator(ABC):
    """
    所有数据生成器的抽象基类
    """

    @abstractmethod
    def generate_data(self, source_data: DataFrame) -> DataFrame:
        """
        @param source_data: 源数据，一般为特征数据feature_data
        @return generated_data: 返回生成的数据
        """
        pass
