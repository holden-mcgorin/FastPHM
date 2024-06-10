class Result:
    """
    轴承预测数据
    """

    def __init__(self, begin_index: int,
                 upper: list = None, mean: list = None, lower: list = None) -> None:
        """
        :param begin_index: 开始预测时的下标
        :param upper: 不确定性区间上界
        :param mean: 预测值（确定性预测）中值或均值（不确定性预测）
        :param lower: 不确定性区间下界
        """
        self.begin_index = begin_index
        self.upper = upper
        self.mean = mean
        self.lower = lower

    def __str__(self) -> str:
        return f"begin_index = {self.begin_index}\nmean = {self.mean}"
