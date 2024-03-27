from rulframework.entity.Bearing import Bearing
from rulframework.predict.evaluator.metric.ABCMetric import ABCMetric


class RUL(ABCMetric):
    @property
    def name(self) -> str:
        return 'RUL'

    def measure(self, bearing: Bearing) -> str:
        total_life = bearing.stage_data.eol_feature
        predict_beginning = bearing.predict_history.begin_index
        return str(total_life - predict_beginning)

