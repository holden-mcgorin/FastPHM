from rulframework.entity.Bearing import Bearing
from rulframework.predict.evaluator.metric.ABCMetric import ABCMetric


class Mean(ABCMetric):
    @property
    def name(self) -> str:
        return 'Mean'

    def measure(self, bearing: Bearing) -> str:
        return str(len(bearing.predict_history.prediction))
    