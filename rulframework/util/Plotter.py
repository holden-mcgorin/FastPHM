import numpy as np
from matplotlib import pyplot as plt

from rulframework.entity.Bearing import Bearing
from rulframework.model.ABCModel import ABCModel
from rulframework.predict.Result import Result
from rulframework.predict.ThresholdTrimmer import ThresholdTrimmer


class Plotter:
    """
    画图器，所有的图片统一由画图器处理
    """
    __FIG_SIZE = (10, 6)  # 图片大小
    __DPI = 200  # 分辨率，默认100
    __COLOR_NORMAL_STAGE = 'green'
    __COLOR_DEGENERATION_STAGE = 'orange'
    __COLOR_FAILURE_STAGE = 'red'
    __COLOR_FAILURE_THRESHOLD = 'darkred'

    def __init__(self):
        raise NotImplementedError("不需要实例化")

    @staticmethod
    def loss(model: ABCModel):
        plt.plot(range(0, len(model.loss)), model.loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

    @staticmethod
    def relative_rul(x, y):
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)
        plt.plot([0, len(x)], [1, 0], color='red')
        plt.scatter(x, y, label='Our proposed model', s=1)
        plt.title('result')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('Relative RUL')
        plt.legend()
        plt.show()

    @staticmethod
    def raw(bearing: Bearing, is_save=False):
        """
        绘画原始振动信号图像
        :param bearing:
        :param is_save: 是否保存图片，默认不保存
        :return:
        """
        if bearing.raw_data is None:
            raise Exception("此轴承原始振动信号变量raw_data为None，请先使用数据加载器加载原始数据赋值给此轴承对象！")

        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)

        if bearing.stage_data is None:
            for key in bearing.raw_data.keys():
                plt.plot(bearing.raw_data[key], label=key)
        else:
            plt.plot(np.arange(bearing.stage_data.fpt_raw + 1), bearing.raw_data[:bearing.stage_data.fpt_raw + 1],
                     label='normal stage', color=Plotter.__COLOR_NORMAL_STAGE)
            plt.plot(
                np.arange(bearing.stage_data.eol_raw - bearing.stage_data.fpt_raw + 1) + bearing.stage_data.fpt_raw,
                bearing.raw_data[bearing.stage_data.fpt_raw:bearing.stage_data.eol_raw + 1],
                label='degeneration stage', color=Plotter.__COLOR_DEGENERATION_STAGE)
            plt.plot(np.arange(len(bearing.raw_data) - bearing.stage_data.eol_raw) + bearing.stage_data.eol_raw,
                     bearing.raw_data[bearing.stage_data.eol_raw:],
                     label='failure stage', color=Plotter.__COLOR_FAILURE_STAGE)

        plt.title(bearing.name + ' Raw Vibration Signals')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('vibration')
        plt.legend()
        if is_save:
            plt.savefig(bearing.name + '.png', dpi=300)
        plt.show()

    @staticmethod
    def __feature(bearing: Bearing):
        for key in bearing.feature_data:
            plt.plot(bearing.feature_data[key], label=key)

    @staticmethod
    def __feature_staged(bearing: Bearing):
        plt.plot(np.arange(bearing.stage_data.fpt_feature + 1),
                 bearing.feature_data[:bearing.stage_data.fpt_feature + 1],
                 label='normal stage', color=Plotter.__COLOR_NORMAL_STAGE)
        plt.plot(
            np.arange(
                bearing.stage_data.eol_feature - bearing.stage_data.fpt_feature + 1) + bearing.stage_data.fpt_feature,
            bearing.feature_data[bearing.stage_data.fpt_feature:bearing.stage_data.eol_feature + 1],
            label='degeneration stage',
            color=Plotter.__COLOR_DEGENERATION_STAGE)
        plt.plot(
            np.arange(len(bearing.feature_data[bearing.stage_data.eol_feature:])) + bearing.stage_data.eol_feature,
            bearing.feature_data[bearing.stage_data.eol_feature:],
            label='failure stage',
            color=Plotter.__COLOR_FAILURE_STAGE)
        # 画失效阈值
        plt.axhline(y=bearing.stage_data.failure_threshold_feature, color=Plotter.__COLOR_FAILURE_THRESHOLD,
                    label='failure threshold')

        # 绘制垂直线表示中间点
        plt.axvline(x=bearing.stage_data.fpt_feature, color='skyblue', linestyle='--')
        plt.axvline(x=bearing.stage_data.eol_feature, color='skyblue', linestyle='--')

        # 获取当前坐标轴对象
        ax = plt.gca()

        # 获取坐标轴的上限和下限
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()  # 获取 y_test 轴的上限和下限

        # 添加标注
        # todo 这里默认特征值为一维的数据
        plt.text(bearing.stage_data.fpt_feature + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]),
                 'FPT', color='black', fontsize=12)
        plt.text(bearing.stage_data.eol_feature + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]),
                 'EoL', color='black', fontsize=12)

    @staticmethod
    def feature(bearing: Bearing):
        """
        绘画轴承特征图
        当轴承包含阶段数据时将绘画轴承的阶段特征图
        当轴承包含预测数据时将绘画轴承的预期曲线
        :return:
        """
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)

        # 当轴承包含阶段数据时将绘画轴承的阶段特征图
        if bearing.stage_data is None:
            Plotter.__feature(bearing)
        else:
            Plotter.__feature_staged(bearing)

        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.gca().add_artist(legend)
        plt.title(bearing.name + ' Feature Graph')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('feature value')
        plt.show()

    @staticmethod
    def __degeneration(bearing: Bearing, is_trim: bool = True):
        if is_trim:
            trimmer = ThresholdTrimmer(bearing.stage_data.failure_threshold_feature)
            bearing.result = trimmer.trim(bearing.result)
        # 画置信区间（不确定性预测）
        if bearing.result.lower is not None and bearing.result.upper is not None:
            plt.fill_between(
                np.arange(len(bearing.result.lower) + 1) + bearing.result.begin_index - 1,
                [float(bearing.feature_data.iloc[
                           bearing.result.begin_index, 0])] + bearing.result.lower,
                [float(bearing.feature_data.iloc[
                           bearing.result.begin_index, 0])] + bearing.result.upper,
                alpha=0.25,
                label='confidence_interval')
        # 画预测值（确定性预测和不确定性预测）
        if bearing.result.mean is not None:
            plt.plot(
                np.arange(len(bearing.result.mean) + 1) + bearing.result.begin_index - 1,
                [float(bearing.feature_data.iloc[bearing.result.begin_index, 0])] +
                bearing.result.mean,
                label='mean')

    @staticmethod
    def degeneration(bearing: Bearing, is_trim: bool = True):
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)
        Plotter.__feature_staged(bearing)
        Plotter.__degeneration(bearing, is_trim=is_trim)

        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.gca().add_artist(legend)
        plt.title(bearing.name + ' Degeneration Trend')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('feature value')
        plt.show()
