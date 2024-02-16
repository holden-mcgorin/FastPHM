from abc import ABC, abstractmethod


class ABCFeatureGenerator(ABC):
    """
    所有特征提取器的基类
    """
    @abstractmethod
    def extract_feature(self, raw_data):
        """
        从原始数据中提取特征
        :return:feature_data
        """
        pass

    def save_data(self):
        pass

    def load_data(self):
        pass
