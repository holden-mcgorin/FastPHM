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

    def predict_till_epoch_uncertainty(self, input_data: list, epoch_num: int,
                                       sampling_num: int = 100, confidence_interval: float = 0.95):
        """
        :param input_data: 初始输入数据
        :param epoch_num: 滚动次数
        :param sampling_num: 采样次数
        :param confidence_interval: 置信区间大小，默认95%
        :return: 预测结果  min_list, mean_list, max_list
        """
        original_input = input_data
        max_list, min_list, mean_list, all_history = [], [], [], []

        # 获取100次采样结果
        for j in range(sampling_num):
            predict_history = []
            input_data = original_input
            len_of_input_data = len(input_data)
            for i in range(epoch_num):
                result = self.predictable.predict(input_data)
                predict_history = predict_history + result
                input_data = input_data + result
                input_data = input_data[-len_of_input_data:]
            all_history.append(predict_history)

        # 计算需要保留的范围
        lower_index = int(sampling_num * ((1 - confidence_interval) // 2))  # 下边界索引
        upper_index = int(sampling_num * confidence_interval + ((1 - confidence_interval) // 2))  # 上边界索引

        for i in range(len(all_history[0])):
            column = []
            for j in range(sampling_num):
                column.append(all_history[j][i])
            # 取置信区间
            sorted_list = sorted(column)
            new_list = sorted_list[lower_index:upper_index]
            # 取区间内的最大值、最小值、平均值
            max_list.append(max(new_list))
            min_list.append(min(new_list))
            mean_list.append(sum(new_list) / len(new_list))
        return min_list, mean_list, max_list
