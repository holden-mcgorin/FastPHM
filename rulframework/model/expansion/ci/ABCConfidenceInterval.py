from abc import ABC, abstractmethod

from rulframework.model.Result import Result


class ABCConfidenceInterval(ABC):
    """
    置信区间计算器
    """

    @abstractmethod
    def __call__(self, result: Result) -> Result:
        """
        :param result: 原始result（只有outputs）
        :return: 包含mean、low、high的result
        """
        raise NotImplementedError
