from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric.ABCMetric import ABCMetric


class Mean(ABCMetric):
    @property
    def name(self) -> str:
        return 'Mean'

    def __call__(self, test_set: Dataset, result: Result) -> str:
        mean = 0
        if result.mean is not None:
            for i in range(len(result.mean)):
                if result.mean[i] > test_set.y[0][-1]:
                    mean = i
                    break
        else:
            for i in range(result.outputs.shape[1]):
                if result.outputs[0][i] > test_set.y[0][-1]:
                    mean = i
                    break
        return str(mean)
