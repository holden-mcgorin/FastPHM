from typing import List

from rulframework.data.dataset.ABCGenerator import ABCGenerator
from rulframework.data.dataset.Dataset import Dataset
from rulframework.entity.Bearing import Bearing


class MultiLabelGenerator(ABCGenerator):
    def __init__(self, generators: List[ABCGenerator]):
        self.generators = generators

    @property
    def name(self):
        """
        :return: None,防止自动添加至生成子标签字典
        """
        return None

    def _generate(self, bearing: Bearing) -> Dataset:
        dataset = self.generators[0].generate(bearing)
        for generator in self.generators[1:]:
            another_dataset = generator.generate(bearing)
            dataset.add_sub_label(generator.name, another_dataset.y)
        return dataset
