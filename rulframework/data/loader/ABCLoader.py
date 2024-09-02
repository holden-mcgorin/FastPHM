from abc import ABC, abstractmethod
from typing import Dict, Union

from pandas import DataFrame

from rulframework.entity.ABCEntity import ABCEntity
from rulframework.system.Logger import Logger


class NameIterator:
    # 用来遍历所有已加载的数据
    def __init__(self, name_list: list):
        self.name_list = name_list
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.name_list):
            result = self.name_list[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration


class ABCLoader(ABC):
    """
    所有数据读取器的抽象基类、
    采用懒加载的方式加载数据
    所有子类必须重写下列方法：
        1. _register
        2. _load
    """

    def __init__(self, root: str):
        """
        获取数据集根目录，确定各个数据项的位置
        :param root: 数据集的根目录
        """
        self._root = root  # 此数据集根目录
        # {数据文件-地址}字典、{实体名称-数据}字典
        self._file_dict, self._entity_dict = self._register(root)

        Logger.debug('\n<< ' + str(self))

    def __str__(self) -> str:
        items = '\n'.join([f"\t{key}, location: {value}" for key, value in self._file_dict.items()])
        return f'Root directory of dataset: {self._root}\n{items}'

    def __iter__(self):
        return NameIterator(list(self._entity_dict.keys()))

    def __call__(self, entity_name, columns: str = None):
        """
        获取实体
        :param entity_name: 实体名称
        :param columns: 列
        :return:
        """
        Logger.info(f'Loading data entity: {entity_name}')
        data_frame = self._load(entity_name)
        entity = self._assemble(entity_name, data_frame, columns)
        Logger.info(f'Successfully loaded data entity: {entity_name}')
        return entity

    @abstractmethod
    def _register(self, root: str) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        """
        生成数据项与其位置的字典
        键：数据项名称
        值：数据项文件目录
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _load(self, entity_name: str) -> DataFrame:
        """
        根据数据项名称从数据集中获取数据
        :param entity_name:数据项名称
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _assemble(self, entity_name: str, raw_data: DataFrame, columns: str = None) -> ABCEntity:
        """
        组装成实体对象
        :param entity_name:
        :return:
        """
        raise NotImplementedError
