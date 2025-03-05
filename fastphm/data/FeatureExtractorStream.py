import pandas as pd
from numpy import ndarray

from fastphm.data.processor.ABCProcessor import ABCProcessor
from fastphm.entity.ABCEntity import ABCEntity


class FeatureExtractorStream:
    """
    流式处理提取特征
    """
    def __init__(self, processors: [ABCProcessor]):
        self.processors = processors

    def __call__(self, entity: ABCEntity):
        data: ndarray = entity.raw_data.values

        for processor in self.processors:
            data = processor(data)

        # 转dataframe(将原始数据的表头复制到特征数据的表头)
        column_names_list = entity.raw_data.columns.tolist()
        result = pd.DataFrame(data, columns=column_names_list)

        entity.feature_data = result
