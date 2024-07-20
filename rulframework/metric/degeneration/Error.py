from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric
from rulframework.metric.degeneration.Mean import Mean
from rulframework.metric.degeneration.RUL import RUL


class Error(ABCMetric):
    @property
    def name(self) -> str:
        return 'Error'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        return str(int(RUL()(test_set, result)) - int(Mean()(test_set, result)))
