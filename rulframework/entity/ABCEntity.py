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
    def __init__(self, name: str,
                 raw_data: DataFrame = None, feature_data: DataFrame = None, stage_data: Stage = None):
        self.name = name  # 此轴承名称
        self.raw_data = raw_data  # 此轴承的原始数据
        self.feature_data = feature_data  # 此轴承的特征数据
        self.stage_data = stage_data

    @property
    def life(self):
        """
        该实体的全寿命时长（单位：秒）
        :return:
        """
        return None

    @property
    def rul(self):
        """
        Remaining Useful Life
        该实体的剩余使用寿命（单位：秒）
        :return:
        """
        return None
