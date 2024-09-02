import numpy as np
import pandas as pd
from numpy import ndarray

from rulframework.data.processor.ABCProcessor import ABCProcessor
from rulframework.entity.ABCEntity import ABCEntity


class FeatureExtractor:
    def __init__(self, processor: ABCProcessor):
        self.processor = processor

    def __call__(self, entity: ABCEntity):
        rawdata: ndarray = entity.raw_data.values
        num_feature = rawdata.shape[1]

        # 第一列应用处理
        first = rawdata[:, 0].reshape(-1)
        result = self.processor(first)
        result = result.reshape((-1, 1))

        # 其他列应用处理并与第一列水平堆叠
        for i in range(1, num_feature):
            processed = self.processor(rawdata[:, i].reshape(-1)).reshape((-1, 1))
            result = np.hstack((result, processed))

        # 转dataframe(将原始数据的表头复制到特征数据的表头)
        column_names_list = entity.raw_data.columns.tolist()
        result = pd.DataFrame(result, columns=column_names_list)

        entity.feature_data = result
