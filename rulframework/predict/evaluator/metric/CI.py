from rulframework.entity.Bearing import Bearing
from rulframework.predict.evaluator.metric.ABCMetric import ABCMetric


class CI(ABCMetric):
    @property
    def name(self) -> str:
        return '95%CI'

    def measure(self, bearing: Bearing) -> str:
        min_list = bearing.result.lower
        max_list = bearing.result.upper
        end = min_list.index(max(min_list))
        begin = max_list.index(max(max_list))
        return f'[{begin}, {end}]'
