from abc import ABC

from rulframework.model.ABCModel import ABCModel


class ABCPredictor(ABC):
    """
    预测器抽象基类
    所有预测器必须继承该基类
    """
    def __init__(self, predictable: ABCModel) -> None:
        self.predictable = predictable

