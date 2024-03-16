from rulframework.entity.Bearing import Bearing
from rulframework.evaluator.metric.ABCMetric import ABCMetric


class CI(ABCMetric):
    @property
    def name(self) -> str:
        return '95%CI'

    def measure(self, bearing: Bearing) -> str:
        min_list = bearing.predict_history.min_list
        max_list = bearing.predict_history.max_list
        end = min_list.index(max(min_list))
        begin = max_list.index(max(max_list))
        return f'[{begin}, {end}]'
