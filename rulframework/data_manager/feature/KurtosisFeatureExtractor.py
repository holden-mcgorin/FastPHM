from pandas import DataFrame

from rulframework.data_manager.feature.ABCFeatureExtractor import ABCFeatureExtractor


class KurtosisFeatureExtractor(ABCFeatureExtractor):
    def extract(self, raw_data: DataFrame) -> DataFrame:
        pass
