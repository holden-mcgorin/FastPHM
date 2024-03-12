class PredictHistory:
    """
    轴承预测数据
    """

    def __init__(self, begin_index: int, prediction: list = None,
                 min_list: list = None, mean_list: list = None, max_list: list = None) -> None:
        """
        :param begin_index: 开始预测时的下标
        :param prediction: 预测结果列表(确定性预测)
        :param min_list: 最小值（不确定性预测）
        :param mean_list: 平均值（不确定性预测）
        :param max_list: 最大值（不确定性预测）
        """
        self.begin_index = begin_index
        self.prediction = prediction
        self.min_list = min_list
        self.mean_list = mean_list
        self.max_list = max_list

    def __str__(self) -> str:
        return f"begin_index = {self.begin_index}\nprediction = {self.prediction}"