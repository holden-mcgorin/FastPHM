import numpy as np
from pandas import DataFrame

from rulframework.data.feature.ABCFeatureExtractor import ABCFeatureExtractor


class RMSFeatureExtractor(ABCFeatureExtractor):
    def __init__(self, span: int):
        """
        :param span:用于计算RMS的区间大小
        """
        self.span = span

    def extract(self, raw_data) -> DataFrame:
        feature_values = np.empty((1, raw_data.shape[1]))
        for i in range(0, len(raw_data) - self.span + 1, self.span):
            window = raw_data[i:i + self.span]
            rms = np.sqrt(np.mean(window ** 2))
            feature_values = np.vstack((feature_values, rms))
        feature_values = feature_values[1:, :]
        feature_values = DataFrame(feature_values, columns=raw_data.columns)
        return feature_values
