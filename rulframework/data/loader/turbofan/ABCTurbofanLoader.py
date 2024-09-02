from abc import ABC, abstractmethod
from typing import Dict, Union

from pandas import DataFrame

from rulframework.data.loader.ABCLoader import ABCLoader
from rulframework.entity.Turbofan import Turbofan


class ABCTurbofanLoader(ABCLoader, ABC):
    def __call__(self, entity_name, columns_to_drop: [int] = None) -> Turbofan:
        return super().__call__(entity_name, columns_to_drop)
