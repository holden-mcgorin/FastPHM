from abc import abstractmethod, ABC

from rulframework.data.dataset.Dataset import Dataset
from rulframework.predict.Result import Result


class ABCEnd2EndMetric(ABC):
    """
    所有评价指标的抽象父类
    所有子类必须完成下列功能
    1. 返回评价指标的名称
    2. 完成评价指标的具体计算方法，返回评价结果
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        定义此评价指标的名称
        :return: 此评价指标的名称
        """
        raise NotImplementedError

    @abstractmethod
    def _measure(self, test_set: Dataset, result: Result) -> str:
        """
        此评价指标的计算方法
        :return: 评价指标字符串（数字、区间、百分比...）
        """
        raise NotImplementedError

    def measure(self, test_set: Dataset, result: Result) -> str:
        # 验证输入的合法性
        sample_num = test_set.x.shape[0]
        if sample_num != result.mean.shape[0]:
            raise Exception(f'测试样本量：{sample_num}与测试结果数量：{result.mean.shape[0]} 不匹配')

        return self._measure(test_set, result)
