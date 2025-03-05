"""
顶级类：Bearing
辅助类：BearingStage
"""
from enum import Enum
from typing import List
from pandas import DataFrame

from fastphm.entity.ABCEntity import ABCEntity, Stage


class Fault(Enum):
    """
    轴承故障类型枚举
    """
    NC = 'Normal Condition'
    OF = 'Outer Race Fault'
    IF = 'Inner Race Fault'
    CF = 'Cage Fault'
    BF = 'Ball Fault'

    def __str__(self):
        return self.name
        # return self.value


class Bearing(ABCEntity):
    """
    轴承对象
    适用数据集：PHM2012、XJTU-SY
    """

    def __init__(self, name: str,
                 span: int = None, continuum: int = None, frequency: int = None, fault_type: List[Fault] = None,
                 raw_data: DataFrame = None, feature_data: DataFrame = None, stage_data: Stage = None):
        super().__init__(name, raw_data, feature_data)
        self.stage_data = stage_data  # 此轴承的全寿命阶段划分数据

        self.frequency = frequency  # 此轴承的采样频率
        self.continuum = continuum  # 此轴承的连续采样区间大小
        self.span = span  # 此轴承连续采样代表的时间
        self.fault_type = fault_type  # 故障类型

    def __str__(self) -> str:
        # 生成故障描述
        fault_str = 'fault: "'
        if self.fault_type is not None:
            for fault in self.fault_type:
                fault_str += fault.value + '; '
        else:
            fault_str += 'unknown'
        fault_str += '"'

        frequency_str = 'frequency: "' + str(self.frequency) + 'Hz"'
        continuum = 'continuum: ' + str(self.continuum) + '"'
        span_str = 'window_size: "' + str(self.span) + 's"'

        return self.name + ',  ' + ", ".join([frequency_str, continuum, span_str]) + '  ' + fault_str

    @property
    def life(self):
        """
        轴承的全寿命时长（单位：秒）
        :return:
        """
        return self.raw_data.shape[0] / self.continuum * self.span

    @property
    def rul(self):
        """
        根据FPT计算的轴承的RUL（单位：秒）
        :return:
        """
        return (self.raw_data.shape[0] - self.stage_data.fpt_raw) / self.continuum * self.span
