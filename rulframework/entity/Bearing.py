"""
顶级类：Bearing
辅助类：BearingStage
"""
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from rulframework.predict.PredictHistory import PredictHistory


class BearingStage:
    """
    轴承阶段数据
    """

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


class FaultType(Enum):
    OUTER = 'Outer Race'
    INNER = 'Inner Race'
    CAGE = 'Cage'
    BALL = 'Ball'


class Bearing:
    """
    轴承对象
    """

    FIG_SIZE = (10, 6)  # 图片大小
    DPI = 200  # 分辨率，默认100
    COLOR_NORMAL_STAGE = 'green'
    COLOR_DEGENERATION_STAGE = 'orange'
    COLOR_FAILURE_STAGE = 'red'
    COLOR_FAILURE_THRESHOLD = 'darkred'

    def __init__(self, name: str, span: int = None, fault_type: list = None,
                 raw_data: DataFrame = None, feature_data: DataFrame = None, train_data: DataFrame = None,
                 stage_data: BearingStage = None, predict_history: PredictHistory = None):
        self.name = name  # 此轴承名称
        self.span = span  # 此轴承连续采样的区间大小
        self.fault_type = fault_type  # 故障类型
        self.raw_data = raw_data  # 此轴承的原始数据
        self.feature_data = feature_data  # 此轴承的特征数据
        self.train_data = train_data  # 此轴承用于训练模型的数据
        self.stage_data = stage_data  # 此轴承的全寿命阶段划分数据
        self.predict_history = predict_history  # 此轴承的RUL预测数据

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

        plt.figure(figsize=self.FIG_SIZE, dpi=self.DPI)

        if self.stage_data is None:
            for key in self.raw_data.keys():
                plt.plot(self.raw_data[key], label=key)
        else:
            plt.plot(np.arange(self.stage_data.fpt_raw + 1), self.raw_data[:self.stage_data.fpt_raw + 1],
                     label='normal stage', color=self.COLOR_NORMAL_STAGE)
            plt.plot(np.arange(self.stage_data.eol_raw - self.stage_data.fpt_raw + 1) + self.stage_data.fpt_raw,
                     self.raw_data[self.stage_data.fpt_raw:self.stage_data.eol_raw + 1],
                     label='degeneration stage', color=self.COLOR_DEGENERATION_STAGE)
            plt.plot(np.arange(len(self.raw_data) - self.stage_data.eol_raw) + self.stage_data.eol_raw,
                     self.raw_data[self.stage_data.eol_raw:],
                     label='failure stage', color=self.COLOR_FAILURE_STAGE)

        plt.title(self.name + ' Raw Vibration Signals')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('vibration')
        plt.legend()
        if is_save:
            plt.savefig(self.name + '.png', dpi=300)
        plt.show()

    def plot_feature(self):
        """
        绘画轴承特征图
        当轴承包含阶段数据时将绘画轴承的阶段特征图
        当轴承包含预测数据时将绘画轴承的预期曲线
        :return:
        """
        plt.figure(figsize=self.FIG_SIZE, dpi=self.DPI)

        # 当轴承包含阶段数据时将绘画轴承的阶段特征图
        if self.stage_data is None:
            for key in self.feature_data:
                plt.plot(self.feature_data[key], label=key)

        if self.stage_data is not None:
            plt.plot(np.arange(self.stage_data.fpt_feature + 1), self.feature_data[:self.stage_data.fpt_feature + 1],
                     label='normal stage', color=self.COLOR_NORMAL_STAGE)
            plt.plot(
                np.arange(self.stage_data.eol_feature - self.stage_data.fpt_feature + 1) + self.stage_data.fpt_feature,
                self.feature_data[self.stage_data.fpt_feature:self.stage_data.eol_feature + 1],
                label='degeneration stage',
                color=self.COLOR_DEGENERATION_STAGE)
            plt.plot(np.arange(len(self.feature_data[self.stage_data.eol_feature:])) + self.stage_data.eol_feature,
                     self.feature_data[self.stage_data.eol_feature:],
                     label='failure stage',
                     color=self.COLOR_FAILURE_STAGE)
            # 画失效阈值
            plt.axhline(y=self.stage_data.failure_threshold_feature, color=self.COLOR_FAILURE_THRESHOLD, linestyle='-',
                        label='failure threshold')

            # 绘制垂直线表示中间点
            plt.axvline(x=self.stage_data.fpt_feature, color='skyblue', linestyle='--')
            plt.axvline(x=self.stage_data.eol_feature, color='skyblue', linestyle='--')

            # 获取当前坐标轴对象
            ax = plt.gca()

            # 获取坐标轴的上限和下限
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()  # 获取 y_test 轴的上限和下限

            # 添加标注
            # todo 这里默认特征值为一维的数据
            plt.text(self.stage_data.fpt_feature + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]),
                     'FPT', color='black', fontsize=12)
            plt.text(self.stage_data.eol_feature + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]),
                     'EoL', color='black', fontsize=12)

        # 当轴承包含预测数据时将绘画轴承的预期曲线
        if self.predict_history is not None:
            # 画置信区间（不确定性预测）
            if self.predict_history.lower is not None and self.predict_history.upper is not None:
                plt.fill_between(np.arange(len(self.predict_history.lower) + 1) + self.predict_history.begin_index - 1,
                                 [float(self.feature_data.iloc[
                                            self.predict_history.begin_index, 0])] + self.predict_history.lower,
                                 [float(self.feature_data.iloc[
                                            self.predict_history.begin_index, 0])] + self.predict_history.upper,
                                 alpha=0.25,
                                 label='confidence_interval')
            # 画预测值（确定性预测和不确定性预测）
            if self.predict_history.prediction is not None:
                plt.plot(np.arange(len(self.predict_history.prediction) + 1) + self.predict_history.begin_index - 1,
                         [float(self.feature_data.iloc[self.predict_history.begin_index, 0])] +
                         self.predict_history.prediction,
                         label='prediction')

        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.gca().add_artist(legend)
        plt.title(self.name + ' Feature Graph')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('feature value')
        plt.show()
