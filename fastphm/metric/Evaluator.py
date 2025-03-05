from typing import List

import numpy as np
import pandas as pd

from fastphm.data.Dataset import Dataset
from fastphm.model.Result import Result
from fastphm.metric import ABCMetric
from fastphm.system.Logger import Logger


class Evaluator:
    """
    指标评价器
    先使用add_metric添加需要的指标
    再调用evaluate计算所有的指标
    """

    def __init__(self) -> None:
        # 用于保存评价指标
        self.metrics: List[ABCMetric] = []

        # 用于保存历史数据
        self.history: [[]] = []  # 记录评估历史，用于求平均
        self.sample_size: [] = []  # 记录每次评估的样本数，用于求加权平均
        self.entity_name: [] = []  # 记录每次评估的对象名称

    def __call__(self, test_set: Dataset, result: Result) -> {}:
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

        evaluation = {}
        self.sample_size.append(len(test_set))
        self.entity_name.append(test_set.name)
        string = f'<< Performance Evaluation of {test_set.name}:'
        for index, metric in enumerate(self.metrics):
            try:
                v = metric.value(test_set, result)
                evaluation[metric.name] = v
                string = string + f'\n\t{metric.name}: {metric.format(v)}'
                self.history[index].append(v)
            except NotImplementedError:  # 兼容退化量评价指标
                v = metric(test_set, result)
                evaluation[metric.name] = v
                string = string + f'\n\t{metric.name}: {v}'

        Logger.info('\n' + string)
        return evaluation

    def add(self, *args: ABCMetric) -> None:
        """
        添加评价指标
        :param args:
        :return:
        """
        for arg in args:
            self.metrics.append(arg)
            self.history.append([])

    def clear_metrics(self):
        self.metrics = []

    def clear_history(self):
        self.history = []  # 记录评估历史，用于求平均
        self.sample_size = []  # 记录每次评估的样本数，用于求加权平均
        self.entity_name = []  # 记录每次评估的对象名称
        for i in range(len(self.metrics)):
            self.history.append([])

    def avg_history(self) -> None:
        if len(self.history[0]) == 0:
            Logger.warning('Evaluator empty history')
            return

        string = f'<< Avg Performance Evaluation:'
        total_size = sum(self.sample_size)
        for index, metric in enumerate(self.metrics):
            # 计算加权平均
            weighted_sum = sum(v * w for v, w in zip(self.history[index], self.sample_size))
            e = metric.format(weighted_sum / total_size)
            string = string + f'\n\t{metric.name}: {e}'

        Logger.info('\n' + string)

    def __avg(self, metric_index):
        # 计算加权平均
        total_size = sum(self.sample_size)
        weighted_sum = sum(v * w for v, w in zip(self.history[metric_index], self.sample_size))
        return weighted_sum / total_size

    @property
    def dataframe(self):
        column_names = [metric.name for metric in self.metrics]
        row_names = self.entity_name.copy()
        data = np.array(self.history).T
        # 最后一行计算加权平均
        row_names.append('avg')
        avg_row = []
        for i in range(len(self.metrics)):
            avg = self.__avg(i)
            avg_row.append(avg)
        last_row = np.array(avg_row).reshape(1, -1)
        data = np.vstack((data, last_row))
        return pd.DataFrame(data, index=row_names, columns=column_names)
