from typing import List

from rulframework.data.labeler.ABCLabeler import ABCLabeler
from rulframework.data.Dataset import Dataset
from rulframework.entity.Bearing import Bearing


class MultiLabeler(ABCLabeler):
    def __init__(self, generators: List[ABCLabeler]):
        self.generators = generators

    @property
    def name(self):
        """
        :return: None,防止自动添加至生成子标签字典
        """
        return None

    def _label(self, bearing: Bearing) -> Dataset:
        dataset = self.generators[0].__call__(bearing)
        for generator in self.generators[1:]:
            another_dataset = generator.__call__(bearing)
            dataset.add_sub_label(generator.name, another_dataset.y)
        return dataset
