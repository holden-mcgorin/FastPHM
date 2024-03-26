import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import kurtosis

from rulframework.data.feature.ABCFeatureExtractor import ABCFeatureExtractor


class KurtosisFeatureExtractor(ABCFeatureExtractor):
    def __init__(self, span: int):
        """
        :param span:用于计算的区间大小
        """
        self.span = span

    def extract(self, raw_data: DataFrame) -> DataFrame:
        feature_values = pd.DataFrame()
        # raw_data = raw_data.values
        for i in range(0, len(raw_data) - self.span + 1, self.span):
            window = raw_data[i:i + self.span]

            data_kurtosis = kurtosis(window) + 3
            # mean = np.mean(window)
            # n = len(window)
            # variance = np.var(window)
            # std = np.sqrt(variance)
            # total = 0
            # for j in window:
            #     total += ((float(j) - float(mean)) / std) ** 4
            # data_kurtosis = total / n

            feature_values = feature_values.append(pd.DataFrame(data_kurtosis), ignore_index=True)
        return feature_values
