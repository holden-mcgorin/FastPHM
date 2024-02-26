from abc import ABC, abstractmethod

from pandas import DataFrame


class ABCDataGenerator(ABC):
    """
    所有数据生成器的抽象基类
    """
    @abstractmethod
    def generate_data(self, item_name) -> DataFrame:
        pass
