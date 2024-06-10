from typing import List

from rulframework.data.dataset.Dataset import Dataset
from rulframework.predict.Result import Result
from rulframework.predict.evaluator.end2end_metric.ABCEnd2EndMetric import ABCEnd2EndMetric


class End2EndEvaluator:
    """
    指标评价器
    先使用add_metric添加需要的指标
    再调用evaluate计算所有的指标
    """

    def __init__(self) -> None:
        self.metrics: List[ABCEnd2EndMetric] = []

    def __str__(self) -> str:
        return super().__str__()

    def add_metric(self, *args: ABCEnd2EndMetric) -> None:
        """
        添加评价指标
        :param args:
        :return:
        """
        for arg in args:
            self.metrics.append(arg)

    def evaluate(self, test_set: Dataset, result: Result) -> None:
        """
        根据已经添加的评价指标开始计算
        :param result:
        :param test_set:
        :return:
        """
        print(f'轴承{test_set.name}的预测结果评价：')
        for metric in self.metrics:
            print(f'  {metric.name}： {metric.measure(test_set, result)}')
