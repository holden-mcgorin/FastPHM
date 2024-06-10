from rulframework.predict.Result import Result


class ThresholdTrimmer:
    """
    阈值裁剪器
    1. 将预测的结果序列进行裁剪
    2. 去掉超过阈值的部分
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def trim(self, result: Result) -> Result:
        rectified_predict_history = Result(result.begin_index)

        # 修正预测值
        if result.mean is not None:
            prediction = result.mean.copy()
            threshold_index_mean = 0
            mean_flag = False
            length = len(prediction)
            for i in range(length):
                if not mean_flag and prediction[i] > self.threshold:
                    threshold_index_mean = i
                    mean_flag = True
                    break
            if mean_flag:
                for i in range(threshold_index_mean + 1, length):
                    del prediction[threshold_index_mean + 1]
                prediction[-1] = self.threshold
            rectified_predict_history.mean = prediction

        # 修正不确定性区间
        if (result.lower is not None
                and result.upper is not None):
            lower = result.lower.copy()
            upper = result.upper.copy()

            # 检查超过开始阈值的下标
            threshold_index_min, threshold_index_max = 0, 0
            min_flag, max_flag = False, False  # 是否获取到超过阈值的起始下标
            length = len(upper)
            for i in range(length):
                if not max_flag and upper[i] > self.threshold:
                    threshold_index_max = i
                    max_flag = True
                if not min_flag and lower[i] > self.threshold:
                    threshold_index_min = i
                    min_flag = True

            # 开始修正
            if max_flag:
                for i in range(threshold_index_max, length):
                    upper[i] = self.threshold
            if min_flag:
                for i in range(threshold_index_min + 1, length):
                    del lower[threshold_index_min + 1]
                    del upper[threshold_index_min + 1]
                lower[-1] = self.threshold

            rectified_predict_history.lower = lower
            rectified_predict_history.upper = upper
        return rectified_predict_history
