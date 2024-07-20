from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric


class MSE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MSE'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        y_hat = result.outputs.reshape(-1)
        y = test_set.y.reshape(-1)
        y_hat, y = self.trim_lists(y_hat, y)

        n = len(y)
        squared_errors = [(y[i] - y_hat[i]) ** 2 for i in range(n)]
        mse = sum(squared_errors) / n
        return f"{mse:.4f}"

    @staticmethod
    def trim_lists(list1, list2):
        if len(list1) < len(list2):
            list2 = list2[:len(list1)]
        elif len(list1) > len(list2):
            list1 = list1[:len(list2)]
        return list1, list2
