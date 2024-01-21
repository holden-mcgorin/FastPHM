from abc import ABC, abstractmethod


class FeatureGenerator(ABC):
    """
    所有特征提取器的基类
    """
    @abstractmethod
    def extract_feature(self):
        """
        从原始数据中提取特征
        :return:
        """
        pass

    @abstractmethod
    def save_feature_data(self):
        """
        保存特征数据文件
        :return:
        """
        pass

    @abstractmethod
    def load_feature_data(self):
        """
        加载特征数据文件
        :return:
        """
        pass
