from abc import ABC, abstractmethod

from rulframework.entity import Bearing


class ABCDataLoader(ABC):
    """
    所有数据读取器的抽象基类
    """

    def __init__(self, root_dir):
        """
        加载轴承目录，确定各个轴承数据的位置
        """
        self.root_dir = root_dir  # 此轴承数据集根目录
        self.item_dict = {}  # 单个数据文件夹位置字典
        pass

    @abstractmethod
    def load_data(self, item_name):
        pass

