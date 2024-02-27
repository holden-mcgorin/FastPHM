from abc import ABC, abstractmethod

from pandas import DataFrame


class ABCPredictable(ABC):
    """
    预测器的内核
    所有预测器必须聚合一个ABCPredictable
    模型必须实现ABCPredictable中的predict抽象方法
    使预测器与模型能够规范接口联合使用
    """

    @abstractmethod
    def predict(self, input_data: list):
        pass
