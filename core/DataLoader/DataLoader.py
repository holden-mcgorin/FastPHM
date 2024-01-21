from abc import ABC, abstractmethod

from core.Bearing import Bearing


class DataLoader(ABC):
    """
    所有数据读取器的抽象基类
    """

    def __init__(self, root_dir):
        """
        加载轴承目录，确定各个轴承数据的位置
        """
        self.root_dir = root_dir  # 此轴承数据集根目录
        self.bearing_dict = {}  # 单个轴承数据文件夹位置字典
        pass

    @abstractmethod
    def load_bearing(self, bearing_name) -> Bearing:
        pass

