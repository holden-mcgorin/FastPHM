from abc import ABC, abstractmethod

from fastphm.data import Dataset
from fastphm.model.Result import Result


class ABCTester(ABC):
    """
    所有测试器的接口
    """

    def __init__(self, config: dict = None):
        # 初始化训练配置
        self.config = config if config else {}

    @abstractmethod
    def test(self,
             model,
             test_set: Dataset) -> Result:
        pass

    # @abstractmethod
    # def _on_test_begin(self):
    #     pass
    #
    # @abstractmethod
    # def _on_test(self):
    #     pass
