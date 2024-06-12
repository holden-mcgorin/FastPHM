from scipy.stats import mode

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from rulframework.data.dataset.Dataset import Dataset
from rulframework.entity.Bearing import Bearing
from rulframework.model.ABCModel import ABCModel
from rulframework.predict.Result import Result
from rulframework.util.ThresholdTrimmer import ThresholdTrimmer


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
    def __staged(data, fpt, eol):
        """
        将数据分阶段
        :param data: 数据
        :param fpt:
        :param eol:
        :return:
        """
        plt.plot(np.arange(fpt + 1), data[:fpt + 1], label='normal stage', color=Plotter.__COLOR_NORMAL_STAGE)
        plt.plot(np.arange(eol - fpt + 1) + fpt, data[fpt:eol + 1], label='degeneration stage',
                 color=Plotter.__COLOR_DEGENERATION_STAGE)
        plt.plot(np.arange(len(data[eol:])) + eol, data[eol:], label='failure stage',
                 color=Plotter.__COLOR_FAILURE_STAGE)

    @staticmethod
    def raw(bearing: Bearing, is_staged=True, is_save=False):
        """
        绘画原始振动信号图像
        :param is_staged:
        :param bearing:
        :param is_save: 是否保存图片，默认不保存
        :return:
        """
        if bearing.raw_data is None:
            raise Exception("此轴承原始振动信号变量raw_data为None，请先使用数据加载器加载原始数据赋值给此轴承对象！")

        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)

        if bearing.stage_data is None or not is_staged:
            for key in bearing.raw_data.keys():
                plt.plot(bearing.raw_data[key], label=key)
        else:
            fpt = bearing.stage_data.fpt_raw
            eol = bearing.stage_data.eol_raw
            data = bearing.raw_data
            Plotter.__staged(data, fpt, eol)

        plt.title(bearing.name + ' Raw Vibration Signals')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('vibration')
        plt.legend()
        if is_save:
            plt.savefig(bearing.name + ' Raw Vibration Signals')
        plt.show()

    @staticmethod
    def __feature(bearing: Bearing, is_staged=True):
        fpt = bearing.stage_data.fpt_feature
        eol = bearing.stage_data.eol_feature
        data = bearing.feature_data

        if bearing.stage_data is None or not is_staged:
            for key in bearing.feature_data:
                plt.plot(bearing.feature_data[key], label=key)
        else:
            Plotter.__staged(data, fpt, eol)
            # 画失效阈值
            plt.axhline(y=bearing.stage_data.failure_threshold_feature, color=Plotter.__COLOR_FAILURE_THRESHOLD,
                        label='failure threshold')

            # 绘制垂直线表示中间点
            plt.axvline(x=fpt, color='skyblue', linestyle='--')
            plt.axvline(x=eol, color='skyblue', linestyle='--')

            # 获取当前坐标轴对象
            ax = plt.gca()
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()  # 获取 y_test 轴的上限和下限

            # 添加标注
            # todo 这里默认特征值为一维的数据
            plt.text(fpt + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]), 'FPT', color='black', fontsize=12)
            plt.text(eol + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]), 'EoL', color='black', fontsize=12)

    @staticmethod
    def feature(bearing: Bearing, is_staged=True, is_save=False):
        """
        绘画轴承特征图，当存在阶段数据且设为True时画阶段特征图
        :return:
        """
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)

        Plotter.__feature(bearing, is_staged)

        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.gca().add_artist(legend)
        plt.title(bearing.name + ' Feature Graph')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('feature value')
        if is_save:
            plt.savefig(bearing.name + ' Feature Graph')
        plt.show()

    @staticmethod
    def __degeneration_rul(bearing: Bearing, is_trim: bool = True):
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
    def degeneration_rul(bearing: Bearing, is_trim: bool = True, is_staged: bool = True):
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)

        Plotter.__feature(bearing)
        Plotter.__degeneration_rul(bearing, is_trim=is_trim)

        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.gca().add_artist(legend)
        plt.title(bearing.name + ' Degeneration Trend')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('feature value')
        plt.show()

    @staticmethod
    def end2end_rul(test_set: Dataset, result: Result):
        x = test_set.z.reshape(-1) / 60
        y = result.mean.reshape(-1)

        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)
        plt.plot([0, max(x)], [1, 0], color='red')
        plt.scatter(x, y, label='Our proposed model', s=1)
        plt.title(f'RUL prediction result of {test_set.name}')
        plt.xlabel('Time (min)')
        plt.ylabel('Relative RUL')
        plt.legend()
        plt.show()

    @staticmethod
    def fault_during_time(test_set, result, interval=1):
        # todo 存在bug，当interval不能被行数整除时会报错
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)
        plt.ylim(-0.4, test_set.y.shape[1] - 0.6)

        x = test_set.z / 60
        y = np.argmax(result.mean, axis=1)  # 找出每行最大值的下标
        y = y.reshape(-1, 1)

        # 将数据按时间排序
        xy = np.hstack((x, y))
        last_column_index = xy[:, 0]
        sorted_indices = np.argsort(last_column_index)
        # 重新排列矩阵的行
        xy = xy[sorted_indices]

        x = xy[:, 0]
        y = xy[:, 1]

        x = x.reshape(-1, interval)
        x = np.mean(x, axis=1).reshape(-1)
        y = y.reshape(-1, interval)
        # 找出每行出现最多次的元素构成新的列向量
        y = np.apply_along_axis(lambda l: mode(l)[0], axis=1, arr=y).reshape(-1)

        plt.scatter(x, y, label='Our proposed model', s=1)

        plt.title(f'Fault Type Prediction Result of {test_set.name}')
        plt.xlabel('Time (min)')
        plt.ylabel('Predicted Fault Label')
        # plt.legend()
        plt.show()

    @staticmethod
    def fault_prediction_heatmap(test_set, result):
        """
        故障诊断热图（混淆矩阵图）
        :return:
        """
        y_true = np.argmax(test_set.y, axis=1)
        y_pred = np.argmax(result.mean, axis=1)  # 找出每行最大值的下标

        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)
        # 数据
        labels = list(Bearing.FaultType.__members__)

        # 获取类别的数量
        num_classes = len(labels)
        classes = np.arange(num_classes)

        # 构建混淆矩阵
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # 遍历每个样本，增加对应位置的计数值
        for i in range(len(y_true)):
            true_class_index = np.where(classes == y_true[i])[0][0]
            pred_class_index = np.where(classes == y_pred[i])[0][0]
            conf_matrix[true_class_index][pred_class_index] += 1

        # 计算每一行的总和
        row_sums = conf_matrix.sum(axis=1, keepdims=True)

        # 将每个元素除以相应行的总和，并乘以 100
        conf_matrix_percent = np.zeros_like(conf_matrix, dtype=float)
        for i in range(num_classes):
            if row_sums[i] != 0:
                conf_matrix_percent[i] = conf_matrix[i] / row_sums[i] * 100

        conf_matrix_percent = conf_matrix_percent.astype(np.int).T

        # 绘制热图
        heatmap = sns.heatmap(conf_matrix_percent, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=labels,
                              yticklabels=labels)

        # 设置标签
        plt.xlabel('True label')
        plt.ylabel('Predicted label')

        # 将y轴文字恢复正常角度
        heatmap.set_yticklabels(labels, rotation=0)

        # 显示图形
        plt.show()
