from abc import ABC, abstractmethod

from pandas import DataFrame


class ABCEoLCalculator(ABC):

    @abstractmethod
    def __call__(self, raw_data: DataFrame, feature_data: DataFrame, scale: int, fpt_raw: int, fpt_feature: int) -> (
            int, int, int, int):
        """
        :param raw_data:
        :param feature_data:
        :param scale:
        :param fpt_raw:
        :param fpt_feature:
        :return: eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature
        """
        raise NotImplementedError
