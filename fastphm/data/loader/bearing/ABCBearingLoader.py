from abc import ABC, abstractmethod
from typing import Dict

from pandas import DataFrame

from fastphm.data.loader.ABCLoader import ABCLoader
from fastphm.entity.ABCEntity import ABCEntity
from fastphm.entity.Bearing import Bearing
from fastphm.system.Logger import Logger


class ABCBearingLoader(ABCLoader):

    def __call__(self, entity_name, columns: str = None) -> Bearing:
        return super().__call__(entity_name, columns)

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

    @property
    def fault_type_dict(self) -> dict:
        """
        可选重写
        如果需要加入故障类型需要重写此方法，反之不需要重写
        :return:
        """
        return dict()

    def _assemble(self, entity_name: str, raw_data: DataFrame, columns: str = None) -> Bearing:
        """
        根据轴承名称从数据集中获取轴承对象
        :param entity_name:数据项名称
        :param columns: 只取指定列数据（水平或垂直信号）
        :return:
        """
        # 生成轴承对象
        bearing = Bearing(entity_name)

        # 赋予原始数据
        # 从数据文件中加载数据（该方法需要在子类中重写）
        raw_data = self._load(entity_name)
        # 如果有column参数则仅取该列数据
        if columns is not None:
            columns_names = raw_data.columns.tolist()
            for name in columns_names:
                if name != columns:
                    raw_data.drop(name, axis=1, inplace=True)
        bearing.raw_data = raw_data

        # 赋予故障类型
        try:
            bearing.fault_type = self.fault_type_dict[entity_name]
        except KeyError:
            pass

        # 赋予连续采样区间、代表时长、采样频率
        bearing.frequency = self.frequency
        bearing.continuum = self.continuum
        bearing.span = self.span

        return bearing
