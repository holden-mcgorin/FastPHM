from enum import Enum
from typing import List

from pandas import DataFrame

from fastphm.entity.ABCEntity import ABCEntity


class Fault(Enum):
    """
    故障类型枚举
    """
    HPC = 'HPC Degradation'
    Fan = 'Fan Degradation'

    def __str__(self):
        return self.name


class Condition(Enum):
    """
    故障类型枚举
    """
    ONE = 'ONE (Sea Level)'
    SIX = 'SIX'
    # SIX的子枚举，即6种工况下更具体的子工况
    SIX_1 = 'condition1'
    SIX_2 = 'condition2'
    SIX_3 = 'condition3'
    SIX_4 = 'condition4'
    SIX_5 = 'condition5'
    SIX_6 = 'condition6'

    def __str__(self):
        return self.name


class Turbofan(ABCEntity):
    """
    适用数据集：C-MAPSS
    """

    def __init__(self, name: str,
                 fault_type: List[Fault] = None, condition: List[Condition] = None,
                 rul: int = None, life: int = None,
                 raw_data: DataFrame = None, feature_data: DataFrame = None):
        super().__init__(name, raw_data, feature_data)
        self.fault_type = fault_type  # 故障类型
        self.condition = condition  # 工况
        self.rul = rul  # 剩余使用寿命
        self.life = life  # 全寿命时长
