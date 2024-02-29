import numpy as np
import pandas as pd
from pandas import DataFrame

from rulframework.data_manager.feature_data.ABCFeatureExtractor import ABCFeatureExtractor


class RMSFeatureExtractor(ABCFeatureExtractor):
    def __init__(self, span: int):
        """
        :param span:用于计算RMS的区间大小
        """
        self.span = span

    def extract(self, raw_data) -> DataFrame:
        feature_values = pd.DataFrame()
        for i in range(0, len(raw_data) - self.span + 1, self.span):
            window = raw_data[i:i + self.span]
            rms = np.sqrt(np.mean(window ** 2))
            feature_values = feature_values.append(rms.to_frame().T, ignore_index=True)
        return feature_values
