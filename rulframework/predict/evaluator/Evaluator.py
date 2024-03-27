from typing import List

from rulframework.entity.Bearing import Bearing
from rulframework.predict.evaluator.metric.ABCMetric import ABCMetric


class Evaluator:
    """
    指标评价器
    先使用add_metric添加需要的指标
    再调用evaluate计算所有的指标
    """
    def __init__(self) -> None:
        self.metrics: List[ABCMetric] = []

    def __str__(self) -> str:
        return super().__str__()

    def add_metric(self, *args: ABCMetric) -> None:
        """
        添加评价指标
        :param args:
        :return:
        """
        for arg in args:
            self.metrics.append(arg)

    def evaluate(self, bearing: Bearing) -> None:
        """
        根据已经添加的评价指标开始计算
        :param bearing:
        :return:
        """
        print(f'轴承{bearing.name}的预测结果评价：')
        for metric in self.metrics:
            print(f'  {metric.name}： {metric.measure(bearing)}')
