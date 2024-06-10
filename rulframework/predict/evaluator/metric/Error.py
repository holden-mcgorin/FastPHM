from rulframework.entity.Bearing import Bearing
from rulframework.predict.evaluator.metric.ABCMetric import ABCMetric


class Error(ABCMetric):
    @property
    def name(self) -> str:
        return 'Error'

    def measure(self, bearing: Bearing) -> str:
        median = len(bearing.result.mean)
        total_life = bearing.stage_data.eol_feature
        predict_beginning = bearing.result.begin_index
        rul = total_life - predict_beginning
        return str(median - rul)
