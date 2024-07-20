from abc import ABC, abstractmethod
from typing import Dict

from pandas import DataFrame

from rulframework.entity.Bearing import Bearing
from rulframework.system.Logger import Logger


class NameIterator:
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
        self._root = root  # 此数据集根目录
        self._item_dict = self._build_item_dict(root)  # 单个数据文件夹位置字典
        Logger.debug('\n' + str(self))

    def __iter__(self):
        return NameIterator(list(self._item_dict.keys()))

    def __str__(self) -> str:
        items = '\n'.join([f"  {key}，位置: {value}" for key, value in self._item_dict.items()])
        return f'>> 数据集位置：{self._root}\n' \
               f'{items}'

    @property
    def fault_type_dict(self) -> dict:
        """
        可选重写
        如果需要加入故障类型需要重写此方法，反之不需要重写
        :return:
        """
        return dict()

    @property
    @abstractmethod
    def frequency(self) -> int:
        """
        :return: 采样频率（单位：Hz）
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def continuum(self) -> int:
        """
        :return: 该数据集每次连续采样的样本数量（单位：个）
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def span(self) -> int:
        """
        :return: 每次采样代表的时长=每次采样的时长+每次未采样的时长（单位：秒）
        """
        raise NotImplementedError

    @abstractmethod
    def _build_item_dict(self, root) -> Dict[str, str]:
        """
        生成数据项与其位置的字典
        键：数据项名称
        值：数据项文件目录
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _load_raw_data(self, item_name) -> DataFrame:
        """
        根据数据项名称从数据集中获取数据
        :param item_name:数据项名称
        :return:
        """
        raise NotImplementedError

    def get(self, bearing_name, columns: str = None) -> Bearing:
        """
        根据轴承名称从数据集中获取轴承对象
        :param bearing_name:数据项名称
        :param columns: 只取指定列数据（水平或垂直信号）
        :return:
        """
        Logger.debug(f'正在加载数据项：{bearing_name}')
        # 生成轴承对象
        bearing = Bearing(bearing_name)

        # 赋予原始数据
        # 从数据文件中加载数据（该方法需要在子类中重写）
        raw_data = self._load_raw_data(bearing_name)
        # 如果有column参数则仅取该列数据
        if columns is not None:
            columns_names = raw_data.columns.tolist()
            for name in columns_names:
                if name != columns:
                    raw_data.drop(name, axis=1, inplace=True)
        bearing.raw_data = raw_data

        # 赋予故障类型
        try:
            bearing.fault_type = self.fault_type_dict[bearing_name]
        except KeyError:
            pass

        # 赋予连续采样区间、代表时长、采样频率
        bearing.frequency = self.frequency
        bearing.continuum = self.continuum
        bearing.span = self.span

        Logger.debug(f'成功加载数据项：{bearing_name}')
        return bearing
