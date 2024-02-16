import numpy as np
import pandas as pd

from core.data_manager.feature_data.ABCFeatureGenerator import ABCFeatureGenerator


class RMSFeatureGenerator(ABCFeatureGenerator):
    def __init__(self, span):
        self.span = span

    def extract_feature(self, raw_data):
        feature_values = pd.DataFrame()
        for i in range(0, len(raw_data) - self.span + 1, self.span):
            window = raw_data[i:i + self.span]
            rms = np.sqrt(np.mean(window ** 2))
            feature_values = feature_values.append(rms.to_frame().T, ignore_index=True)
        return feature_values


