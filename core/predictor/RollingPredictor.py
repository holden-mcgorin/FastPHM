from core.predictor.ABCPredictable import ABCPredictable
from core.predictor.ABCPredictor import ABCPredictor


class RollingPredictor(ABCPredictor):
    def __init__(self, predictable: ABCPredictable) -> None:
        super().__init__(predictable)

    def predict_till_epoch(self, input_data: list, epoch_num: int):
        """
        固定滚动次数滚动预测
        :param input_data:初始输入数据
        :param epoch_num:滚动次数
        :return:预测结果list
        """
        predict_history = []
        len_of_input_data = len(input_data)
        for i in range(epoch_num):
            result = self.predictable.predict(input_data)
            predict_history = predict_history + result
            input_data = input_data + result
            input_data = input_data[-len_of_input_data:]
        return predict_history

    def predict_till_threshold(self, input_data, threshold, max_epoch=100):
        """
        滚动预测直到阈值
        :param input_data:初始输入数据
        :param threshold:阈值
        :param max_epoch: 如果一直达不到阈值，需要有一个最大滚动次数
        :return:预测结果list
        """
        predict_history = []
        len_of_input_data = len(input_data)
        reach_threshold = False
        for i in range(max_epoch):
            result = self.predictable.predict(input_data)
            for item in result:
                predict_history.append(item)
                if item > threshold:
                    reach_threshold = True
                    break
            if reach_threshold:
                break
            input_data = input_data + result
            input_data = input_data[-len_of_input_data:]
        return predict_history
