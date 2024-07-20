from typing import List

from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result
from rulframework.metric import ABCMetric
from rulframework.system.Logger import Logger


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

    def add(self, *args: ABCMetric) -> None:
        """
        添加评价指标
        :param args:
        :return:
        """
        for arg in args:
            self.metrics.append(arg)

    def __call__(self, test_set: Dataset, result: Result) -> None:
        """
        根据已经添加的评价指标开始计算
        :param result:
        :param test_set:
        :return:
        """
        # 验证输入的合法性
        sample_num = test_set.x.shape[0]
        if sample_num != result.outputs.shape[0]:
            raise Exception(f'测试样本量：{sample_num}与测试结果数量：{result.outputs.shape[0]} 不匹配')

        string = f'轴承{test_set.name}的预测结果评价：'
        for metric in self.metrics:
            string = string + f'\n  {metric.name}： {metric(test_set, result)}'
        Logger.info('\n' + string)
