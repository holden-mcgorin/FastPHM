from abc import ABC, abstractmethod
from typing import Generator, Dict

from rulframework.entity.Bearing import Bearing


class ABCDataLoader(ABC):
    """
    所有数据读取器的抽象基类、
    所有子类必须重写下列方法：
        1. load
        2. __build_item_dict
    """

    def __init__(self, root: str):
        """
        获取数据集根目录，确定各个数据项的位置
        :param root: 数据集的根目录
        """
        self.root = root  # 此数据集根目录
        self.item_dict = self._build_item_dict(root)  # 单个数据文件夹位置字典
        print('成功登记以下数据项：')
        for key, value in self.item_dict.items():
            print(f"  {key}，位置: {value}")

    @abstractmethod
    def _build_item_dict(self, root) -> Dict[str, str]:
        """
        生成数据项与其位置的字典
        键：数据项名称
        值：数据项文件目录
        :return:
        """
        pass

    @abstractmethod
    def load(self, item_name) -> object:
        """
        根据名称从数据集中获取数据
        :param item_name:数据项名称
        :return:
        """
        pass

    def all_data(self) -> Generator[object, None, None]:
        """
        返回这个数据集的generator
        1. 用 for i in generator 遍历所有数据
        2. 用 next(generator) 逐个获取数据
        :return:
        """
        for item_name in self.item_dict.keys():
            yield self.load(item_name)
