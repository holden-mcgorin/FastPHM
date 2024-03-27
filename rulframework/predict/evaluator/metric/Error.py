from rulframework.entity.Bearing import Bearing
from rulframework.predict.evaluator.metric.ABCMetric import ABCMetric


class Error(ABCMetric):
    @property
    def name(self) -> str:
        return 'Error'

    def measure(self, bearing: Bearing) -> str:
        median = len(bearing.predict_history.prediction)
        total_life = bearing.stage_data.eol_feature
        predict_beginning = bearing.predict_history.begin_index
        rul = total_life - predict_beginning
        return str(median - rul)
