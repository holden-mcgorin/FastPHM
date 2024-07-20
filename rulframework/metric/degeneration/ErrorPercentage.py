from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric
from rulframework.metric.degeneration.Error import Error
from rulframework.metric.degeneration.RUL import RUL


class ErrorPercentage(ABCMetric):
    @property
    def name(self) -> str:
        return 'Error percentage'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        error = float(Error()(test_set, result))
        rul = float(RUL()(test_set, result))
        return f"{abs(error / rul) * 100:.1f}%"
