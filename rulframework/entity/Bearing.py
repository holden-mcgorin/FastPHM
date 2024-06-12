"""
顶级类：Bearing
辅助类：BearingStage
"""
from enum import Enum
from typing import List
from numpy import ndarray
from pandas import DataFrame

from rulframework.predict.Result import Result


class BearingStage:
    """
    轴承阶段数据
    """

    def __init__(self, fpt_raw=None, fpt_feature=None,
                 eol_raw=None, eol_feature=None,
                 failure_threshold_raw=None, failure_threshold_feature=None):
        self.fpt_raw = fpt_raw
        self.fpt_feature = fpt_feature
        self.eol_raw = eol_raw
        self.eol_feature = eol_feature
        self.failure_threshold_raw = failure_threshold_raw
        self.failure_threshold_feature = failure_threshold_feature

    def __str__(self) -> str:
        return f"fpt_raw = {self.fpt_raw}, fpt_feature = {self.fpt_feature}, " \
               f"eol_raw = {self.eol_raw}, eol_feature = {self.eol_feature}, " \
               f"failure_threshold_raw = {self.failure_threshold_raw}, " \
               f"failure_threshold_feature = {self.failure_threshold_feature}"


class Bearing:
    """
    轴承对象
    """

    class FaultType(Enum):
        """
        轴承故障类型枚举
        """
        NC = 'Normal Condition'
        OF = 'Outer Race Fault'
        IF = 'Inner Race Fault'
        CF = 'Cage Fault'
        BF = 'Ball Fault'

    def __init__(self, name: str, span: int = None, continuum: int = None, frequency: int = None,
                 fault_type: List[FaultType] = None,
                 raw_data: DataFrame = None, feature_data: DataFrame = None, train_data: ndarray = None,
                 stage_data: BearingStage = None, result: Result = None):
        self.name = name  # 此轴承名称
        self.frequency = frequency  # 此轴承的采样频率
        self.continuum = continuum  # 此轴承的连续采样区间大小
        self.span = span  # 此轴承连续采样代表的时间
        self.fault_type = fault_type  # 故障类型
        self.raw_data = raw_data  # 此轴承的原始数据
        self.feature_data = feature_data  # 此轴承的特征数据
        self.train_data = train_data  # 此轴承用于训练模型的数据
        self.stage_data = stage_data  # 此轴承的全寿命阶段划分数据
        self.result = result  # 此轴承的RUL预测数据

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
        span_str = 'span: "' + str(self.span) + 's"'

        return self.name + ',  ' + ", ".join([frequency_str, continuum, span_str]) + '  ' + fault_str

    @property
    def total_life(self):
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
