from pandas import DataFrame


class Entity:
    """
    备选方案
    """
    def __init__(self, name: str,
                 raw_data: DataFrame = None, feature_data: DataFrame = None):
        self.name = name  # 此轴承名称
        self.raw_data = raw_data  # 此轴承的原始数据
        self.feature_data = feature_data  # 此轴承的特征数据
