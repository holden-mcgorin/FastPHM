class PredictHistory:
    """
    轴承预测数据
    """

    def __init__(self, begin_index: int,
                 upper: list = None, prediction: list = None, lower: list = None) -> None:
        """
        :param begin_index: 开始预测时的下标
        :param upper: 不确定性区间上界
        :param prediction: 预测值（确定性预测）中值或均值（不确定性预测）
        :param lower: 不确定性区间下界
        """
        self.begin_index = begin_index
        self.upper = upper
        self.prediction = prediction
        self.lower = lower

    def __str__(self) -> str:
        return f"begin_index = {self.begin_index}\nprediction = {self.prediction}"
