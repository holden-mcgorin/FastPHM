from abc import ABC, abstractmethod

from rulframework.data.Dataset import Dataset
from rulframework.entity.Bearing import Bearing


class ABCGenerator(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _generate(self, bearing: Bearing) -> Dataset:
        raise NotImplementedError

    def generate(self, bearing: Bearing) -> Dataset:
        # 给数据集的 sub_label 添加信息
        dataset = self._generate(bearing)
        if self.name is not None:
            dataset.sub_label_map[self.name] = [0, dataset.y.shape[1]]

        return dataset
