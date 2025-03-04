from abc import ABC

from pandas import DataFrame


class Stage:
    """
    阶段数据
    """

    def __init__(self, fpt_raw=None, fpt_feature=None, eol_raw=None, eol_feature=None,
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


class ABCEntity(ABC):
    """
    数据实体，如：
        1. 轴承
        2. 发动机
        3. 电池
    数据实体包含三类数据：
        1.原始数据（传感器收集到的数据）
        2.特征数据（预处理后得到的数据，如归一化、RMS、峭度、最大值、最小值……）
        3.阶段数据（阶段划分数据，如正常阶段、退化阶段）
    """
    def __init__(self, name: str,
                 raw_data: DataFrame = None, feature_data: DataFrame = None, stage_data: Stage = None):
        self.name = name  # 此实体名称
        self.raw_data = raw_data  # 此实体的原始数据
        self.feature_data = feature_data  # 此实体的特征数据
        self.stage_data = stage_data  # 此实体的阶段数据
