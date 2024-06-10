from typing import Union

from rulframework.model.ABCModel import ABCModel
from rulframework.predict.predictor.ABCPredictor import ABCPredictor
from rulframework.predict.confidence_interval.ABCCICalculator import ABCCICalculator


class RollingPredictor(ABCPredictor):
    def __init__(self, predictable: ABCModel) -> None:
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
                                       ci_calculator: ABCCICalculator, sampling_num: int = 100, ):
        """
        :param input_data: 初始输入数据
        :param epoch_num: 滚动次数
        :param sampling_num: 采样次数
        :param ci_calculator: 置信区间计算算法
        :return: 预测结果  lower, mean, upper
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

        for i in range(len(all_history[0])):
            column = []
            for j in range(sampling_num):
                column.append(all_history[j][i])
            # 取区间内的最大值、最小值、平均值
            min_value, mean_value, max_value = ci_calculator.calculate(column)
            max_list.append(max_value)
            min_list.append(min_value)
            mean_list.append(mean_value)
        return min_list, mean_list, max_list

    def predict_till_epoch_uncertainty_flat(self, input_data: list, epoch_num: int, threshold: Union[int, float],
                                            ci_calculator: ABCCICalculator, sampling_num: int = 100):
        """
        对 predict_till_epoch_uncertainty 的结果进行修正以画图
        去掉超过失效阈值的图像
        :param threshold:
        :param input_data:
        :param epoch_num:
        :param ci_calculator:
        :param sampling_num:
        :return:
        """
        min_list, mean_list, max_list = self.predict_till_epoch_uncertainty(input_data, epoch_num, ci_calculator,
                                                                            sampling_num)

        return self.fix_till_threshold(min_list, mean_list, max_list, threshold)

    @staticmethod
    def fix_till_threshold(
            raw_min_list: list,
            raw_mean_list: list,
            raw_max_list: list,
            threshold: Union[int, float]) -> (list, list, list):
        """
        修正区间数据，为了生成图片时预测图像不超过失效阈值
        :param raw_min_list: 修正前的 lower
        :param raw_mean_list: 修正前的 mean
        :param raw_max_list: 修正前的 upper
        :param threshold:
        :return: lower, mean, upper
        """
        # 复制列表
        min_list = raw_min_list.copy()
        mean_list = raw_mean_list.copy()
        max_list = raw_max_list.copy()

        # 检查超过开始阈值的下标
        threshold_index_min, threshold_index_mean, threshold_index_max = 0, 0, 0
        min_flag, mean_flag, max_flag = False, False, False  # 是否获取到超过阈值的起始下标
        length = len(max_list)
        for i in range(length):
            if not max_flag and max_list[i] > threshold:
                threshold_index_max = i
                max_flag = True
            if not mean_flag and mean_list[i] > threshold:
                threshold_index_mean = i
                mean_flag = True
            if not min_flag and min_list[i] > threshold:
                threshold_index_min = i
                min_flag = True

        # 开始修正
        if max_flag:
            for i in range(threshold_index_max, length):
                max_list[i] = threshold
        if min_flag:
            for i in range(threshold_index_min, length):
                min_list[i] = threshold
        if mean_flag:
            for i in range(threshold_index_mean + 1, length):
                del mean_list[threshold_index_mean + 1]
            mean_list[-1] = threshold

        return min_list, mean_list, max_list
