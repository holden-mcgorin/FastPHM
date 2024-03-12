from rulframework.entity.PredictHistory import PredictHistory


class ThresholdTrimmer:
    """
    阈值裁剪器
    1. 将预测的结果序列进行裁剪
    2. 去掉超过阈值的部分
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def trim(self, predict_history: PredictHistory) -> PredictHistory:
        fixed_predict_history = PredictHistory(predict_history.begin_index)

        self.__trim_certainty(fixed_predict_history, predict_history)
        self.__trim_uncertainty(fixed_predict_history, predict_history)
        return fixed_predict_history

    def __trim_uncertainty(self, fixed_predict_history, predict_history):
        """
        修正不确定性预测结果
        :param fixed_predict_history:
        :param predict_history:
        :return:
        """
        if (predict_history.min_list is not None
                and predict_history.mean_list is not None
                and predict_history.max_list is not None):
            min_list = predict_history.min_list.copy()
            mean_list = predict_history.mean_list.copy()
            max_list = predict_history.max_list.copy()

            # 检查超过开始阈值的下标
            threshold_index_min, threshold_index_mean, threshold_index_max = 0, 0, 0
            min_flag, mean_flag, max_flag = False, False, False  # 是否获取到超过阈值的起始下标
            length = len(max_list)
            for i in range(length):
                if not max_flag and max_list[i] > self.threshold:
                    threshold_index_max = i
                    max_flag = True
                if not mean_flag and mean_list[i] > self.threshold:
                    threshold_index_mean = i
                    mean_flag = True
                if not min_flag and min_list[i] > self.threshold:
                    threshold_index_min = i
                    min_flag = True

            # 开始修正
            if max_flag:
                for i in range(threshold_index_max, length):
                    max_list[i] = self.threshold
            if min_flag:
                for i in range(threshold_index_min, length):
                    min_list[i] = self.threshold
            if mean_flag:
                for i in range(threshold_index_mean + 1, length):
                    del mean_list[threshold_index_mean + 1]
                mean_list[-1] = self.threshold

            fixed_predict_history.min_list = min_list
            fixed_predict_history.mean_list = mean_list
            fixed_predict_history.max_list = max_list

    def __trim_certainty(self, fixed_predict_history, predict_history):
        """
        修正确定性预测结果
        :param fixed_predict_history:
        :param predict_history:
        :return:
        """
        # todo
        pass
