from abc import ABC, abstractmethod

from fastphm.data.Dataset import Dataset
from fastphm.entity.Bearing import Bearing
from fastphm.model.ABCModel import ABCModel


class ABCExperiment(ABC):
    """
    实验的接口，实现该接口后可以直接调用各种验证方法的类（e.g. 交叉验证）
    """
    @abstractmethod
    def get_dataset(self, bearings: [Bearing], base_bearing: Bearing = None) -> (Dataset, Dataset):
        pass

    @abstractmethod
    def get_model(self, train_set: Dataset) -> ABCModel:
        pass

    @abstractmethod
    def test(self, model, test_set: Dataset) -> float:
        pass
