from abc import ABC, abstractmethod

from rulframework.data.dataset.Dataset import Dataset
from rulframework.entity.Bearing import Bearing


class ABCGenerator(ABC):

    @abstractmethod
    def generate(self, bearing: Bearing) -> Dataset:
        pass
