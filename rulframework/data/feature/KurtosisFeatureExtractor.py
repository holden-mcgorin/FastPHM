from pandas import DataFrame

from rulframework.data.feature.ABCFeatureExtractor import ABCFeatureExtractor


class KurtosisFeatureExtractor(ABCFeatureExtractor):
    def extract(self, raw_data: DataFrame) -> DataFrame:
        pass
