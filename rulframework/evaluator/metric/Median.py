from rulframework.entity.Bearing import Bearing
from rulframework.evaluator.metric.ABCMetric import ABCMetric


class Median(ABCMetric):
    @property
    def name(self) -> str:
        return 'Median'

    def measure(self, bearing: Bearing) -> str:
        return str(len(bearing.predict_history.mean_list))
    