from abc import ABC, abstractmethod


class ABCFPTCalculator(ABC):

    @abstractmethod
    def get_fpt(self, raw_data, feature_data, scale) -> (int, int):
        """
        :param raw_data: 原始振动信号
        :param feature_data: 特征信号
        :param scale: 倍率，原始振动信号/特征信号
        :return: fpt_raw, fpt_feature
        """
        raise NotImplementedError
