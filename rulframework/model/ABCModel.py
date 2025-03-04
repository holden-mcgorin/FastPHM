from abc import ABC, abstractmethod

from numpy import ndarray

from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result


class ABCModel(ABC):
    """
    预测器的内核
    对不同深度学习框架的适配器
    所有预测器必须聚合一个ABCPredictable
    模型必须实现ABCPredictable中的predict抽象方法
    使预测器与模型能够规范接口联合使用
    """

    @abstractmethod
    def __init__(self, name=None, model=None):
        self.name = name
        self.model = model

    @abstractmethod
    def __call__(self, x: ndarray) -> ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def loss(self) -> list:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_param(self) -> int:
        raise NotImplementedError

    def test(self, test_set: Dataset) -> Result:
        return Result(outputs=self(test_set.x), name=self.name)

    @abstractmethod
    def train(self, train_set: Dataset, epochs: int = 100,
              batch_size: int = 128, weight_decay: float = 0,
              criterion=None, optimizer=None):
        raise NotImplementedError
