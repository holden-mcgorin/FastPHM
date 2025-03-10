from abc import ABC, abstractmethod

from fastphm.data.Dataset import Dataset
from fastphm.entity.ABCEntity import ABCEntity


# todo 改名LabelConstructor
class ABCLabeler(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _label(self, entity: ABCEntity) -> Dataset:
        raise NotImplementedError

    def __call__(self, entity: ABCEntity) -> Dataset:
        # 给数据集的 sub_label 添加信息
        dataset = self._label(entity)
        if self.name is not None:
            dataset.sub_label_map[self.name] = [0, dataset.y.shape[1]]

        return dataset
