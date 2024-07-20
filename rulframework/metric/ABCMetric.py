from abc import abstractmethod, ABC

from rulframework.data.Dataset import Dataset
from rulframework.model.Result import Result


class ABCMetric(ABC):
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
    def __call__(self, test_set: Dataset, result: Result) -> str:
        """
        此评价指标的计算方法
        :return: 评价指标字符串（数字、区间、百分比...）
        """
        raise NotImplementedError
