import numpy as np
from pandas import DataFrame

from rulframework.data.stage.fpt.ABCFPTCalculator import ABCFPTCalculator
from rulframework.system.Logger import Logger


class ThreeSigmaFPTCalculator(ABCFPTCalculator):
    def __init__(self, ratio=3, max_consecution=5, consecution_ratio=0.3, healthy_ratio=0.3, min_bound=0.1, max_rms=2):
        """
        :param ratio: 几倍sigma
        :param consecution_ratio: 几次连续出现异常值才判定FPT
        """
        # 默认3倍标准差，即3σ
        self.ratio = ratio
        self.healthy_ratio = healthy_ratio
        self.consecution_ratio = consecution_ratio
        self.min_bound = min_bound
        self.max_consecution = max_consecution
        self.max_rms = max_rms

    def __call__(self, raw_data: DataFrame, feature_data: DataFrame, scale: int) -> (int, int):
        fpt_feature = 0
        feature_data = feature_data.iloc[:, 0]  # todo 这里只取第一列做计算fpt，多列情况不适应

        # 如果取全局计算均值和标准差
        sliced_data = feature_data[:int(len(feature_data) * self.healthy_ratio)]
        mu = np.mean(sliced_data)
        sigma = np.std(sliced_data)
        mu_plus_sigma = mu + self.ratio * sigma + self.min_bound
        # mu_minus_sigma = mu - self.ratio * sigma

        consecution_max = int(self.consecution_ratio * len(feature_data))
        if consecution_max > self.max_consecution:
            consecution_max = self.max_consecution

        # print(f'mu_plus_sigma={mu_plus_sigma}')
        # print(f'consecution_max={consecution_max} ')

        consecution_count = 0
        success = False
        for i in range(len(feature_data)):
            x = feature_data[i]
            if x > mu_plus_sigma or x > self.max_rms:
                consecution_count += 1

                # 两个确认异常标志
                if consecution_count == consecution_max:
                    fpt_feature = i - consecution_max
                    success = True
                    break
                if i == len(feature_data) - 1:
                    fpt_feature = i - consecution_count
                    success = True
                    break
            else:
                consecution_count = 0

        fpt_raw = scale * fpt_feature

        if not success:
            Logger.warning('fail to identify FPT, used default value 0')
        return fpt_raw, fpt_feature
