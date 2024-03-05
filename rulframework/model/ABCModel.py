from abc import ABC, abstractmethod

from pandas import DataFrame


class ABCModel(ABC):
    """
    预测器的内核
    所有预测器必须聚合一个ABCPredictable
    模型必须实现ABCPredictable中的predict抽象方法
    使预测器与模型能够规范接口联合使用
    """
    @abstractmethod
    def train(self, train_data_x: DataFrame, train_data_y: DataFrame, num_epochs: int = 1000):
        pass

    @abstractmethod
    def predict(self, input_data: list) -> list:
        pass

    def predict_uncertainty(self, input_data: list) -> list:
        pass
