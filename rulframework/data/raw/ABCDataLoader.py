from abc import ABC, abstractmethod
from typing import Dict

from pandas import DataFrame

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
        self._root = root  # 此数据集根目录
        self._item_dict = self._build_item_dict(root)  # 单个数据文件夹位置字典
        print(self)

    def __str__(self) -> str:
        items = '\n'.join([f"  {key}，位置: {value}" for key, value in self._item_dict.items()])
        return f'<<<< 数据集位置：{self._root} >>>>\n' \
               f'>> 已成功登记以下数据项：\n' \
               f'{items}'

    @property
    def all(self) -> list:
        return list(self._item_dict.keys())

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
    def span(self) -> int:
        """
        :return: 该数据集连续采样的样本区间大小(每分钟采样的样本数)
        """
        pass

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
    def _load_raw_data(self, item_name) -> DataFrame:
        """
        根据数据项名称从数据集中获取数据
        :param item_name:数据项名称
        :return:
        """
        pass

    def get_bearing(self, bearing_name, columns: str = None) -> Bearing:
        """
        根据轴承名称从数据集中获取轴承对象
        :param bearing_name:数据项名称
        :param columns: 只取指定列数据（水平或垂直信号）
        :return:
        """
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

        # 赋予连续采样区间
        bearing.span = self.span

        return bearing
