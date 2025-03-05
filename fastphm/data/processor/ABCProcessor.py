from abc import ABC, abstractmethod

from numpy import ndarray


class ABCProcessor(ABC):
    """
    所有数据处理器的抽象基类
    作用：将向量转为另一个向量
    """

    @abstractmethod
    def __call__(self, source: ndarray) -> ndarray:
        """
        :param source: 1维ndarray
        :return: 一般是1维ndarray，滑动窗口处理器是2维ndarray
        """
        raise NotImplementedError
