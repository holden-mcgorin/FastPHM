import numpy as np
import pandas as pd
from numpy import ndarray

from rulframework.data.processor.ABCProcessor import ABCProcessor
from rulframework.entity.Bearing import Bearing


class FeatureExtractor:
    def __init__(self, processor: ABCProcessor):
        self.processor = processor

    def extract(self, bearing: Bearing):
        rawdata: ndarray = bearing.raw_data.values
        num_feature = rawdata.shape[1]

        first = rawdata[:, 0].reshape(-1)
        result = self.processor(first)
        result = result.reshape((-1, 1))

        for i in range(1, num_feature):
            processed = self.processor(rawdata[:, i].reshape(-1)).reshape((-1, 1))
            result = np.hstack((result, processed))

        # 转dataframe
        column_names_list = bearing.raw_data.columns.tolist()
        result = pd.DataFrame(result, columns=column_names_list)

        bearing.feature_data = result
