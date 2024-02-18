"""
顶级类：Bearing
辅助类：BearingStage
"""
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


class BearingStage:
    def __init__(self, fpt_raw=None, fpt_feature=None,
                 eol_raw=None, eol_feature=None,
                 failure_threshold_raw=None, failure_threshold_feature=None):
        self.fpt_raw = fpt_raw
        self.fpt_feature = fpt_feature
        self.eol_raw = eol_raw
        self.eol_feature = eol_feature
        self.failure_threshold_raw = failure_threshold_raw
        self.failure_threshold_feature = failure_threshold_feature

    def __str__(self) -> str:
        return f"fpt_raw = {self.fpt_raw}, fpt_feature = {self.fpt_feature}, " \
               f"eol_raw = {self.eol_raw}, eol_feature = {self.eol_feature}, " \
               f"failure_threshold_raw = {self.failure_threshold_raw}, " \
               f"failure_threshold_feature = {self.failure_threshold_feature}"


class Bearing:
    """
    轴承对象
    """

    # 常量，生成的图片大小
    FIG_SIZE = (10, 6)

    def __init__(self, name: str,
                 raw_data: DataFrame = None, feature_data: DataFrame = None, train_data: DataFrame = None,
                 stage_data: BearingStage = None,
                 raw_data_loc: str = None):
        self.name = name
        self.raw_data = raw_data
        self.feature_data = feature_data
        self.train_data = train_data
        self.stage_data = stage_data
        self.raw_data_loc = raw_data_loc

    def __str__(self) -> str:
        return self.name

    def plot_raw(self, is_save=False):
        """
        绘画原始振动信号图像
        :param is_save: 是否保存图片，默认不保存
        :return:
        """
        if self.raw_data is None:
            raise Exception("此轴承原始振动信号变量raw_data为None，请先使用数据加载器加载原始数据赋值给此轴承对象！")

        plt.figure(figsize=self.FIG_SIZE)

        if self.stage_data is None:
            for key in self.raw_data.keys():
                plt.plot(self.raw_data[key], label=key)
        else:
            plt.plot(np.arange(self.stage_data.fpt_raw + 1), self.raw_data[:self.stage_data.fpt_raw + 1],
                     label='normal stage', color='green')
            plt.plot(np.arange(self.stage_data.eol_raw - self.stage_data.fpt_raw + 1) + self.stage_data.fpt_raw,
                     self.raw_data[self.stage_data.fpt_raw:self.stage_data.eol_raw + 1],
                     label='degeneration stage', color='orange')
            plt.plot(np.arange(len(self.raw_data) - self.stage_data.eol_raw) + self.stage_data.eol_raw,
                     self.raw_data[self.stage_data.eol_raw:],
                     label='failure stage', color='red')

        plt.title(self.name + ' Vibration Signals')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('vibration')
        plt.legend()
        if is_save:
            plt.savefig(self.name + '.png', dpi=300)
        plt.show()

    def plot_feature(self):
        plt.figure(figsize=self.FIG_SIZE)

        if self.stage_data is None:
            for key in self.feature_data:
                plt.plot(self.feature_data[key], label=key)
            plt.legend()
        else:
            plt.plot(np.arange(self.stage_data.fpt_feature + 1), self.feature_data[:self.stage_data.fpt_feature + 1],
                     label='normal stage', color='green')
            plt.plot(
                np.arange(self.stage_data.eol_feature - self.stage_data.fpt_feature + 1) + self.stage_data.fpt_feature,
                self.feature_data[self.stage_data.fpt_feature:self.stage_data.eol_feature + 1],
                label='degeneration stage',
                color='orange')
            plt.plot(np.arange(len(self.feature_data[self.stage_data.eol_feature:])) + self.stage_data.eol_feature,
                     self.feature_data[self.stage_data.eol_feature:], label='failure stage', color='red')
            # 画失效阈值
            plt.axhline(y=self.stage_data.failure_threshold_feature, color='red', linestyle='-',
                        label='failure threshold')
            # 绘制垂直线表示中间点
            plt.axvline(x=self.stage_data.fpt_feature, color='skyblue', linestyle='--')
            plt.axvline(x=self.stage_data.eol_feature, color='skyblue', linestyle='--')

            # 添加标注
            # todo 这里默认特征值为一维的数据
            plt.text(self.stage_data.fpt_feature + 2, self.feature_data.iloc[self.stage_data.fpt_feature, 0] + 0.5, 'FPT',
                     color='black', fontsize=12)
            plt.text(self.stage_data.eol_feature - 9, self.feature_data.iloc[self.stage_data.eol_feature, 0] - 0.5, 'EoL',
                     color='black', fontsize=12)

            legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
            plt.gca().add_artist(legend)

        plt.title(self.name + ' Vibration Signals')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('vibration')
        plt.show()
