from abc import ABC

from core.predictor.ABCPredictable import ABCPredictable


class ABCPredictor(ABC):
    """
    预测器抽象基类
    所有预测器必须继承该基类
    """
    def __init__(self, predictable: ABCPredictable) -> None:
        self.predictable = predictable

