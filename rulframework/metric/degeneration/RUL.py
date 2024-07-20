from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric


class RUL(ABCMetric):
    @property
    def name(self) -> str:
        return 'RUL'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        return str(test_set.y.shape[1])
