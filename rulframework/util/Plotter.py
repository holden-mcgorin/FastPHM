from numpy import ndarray
from scipy.stats import mode

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from rulframework.data.dataset.Dataset import Dataset
from rulframework.entity.Bearing import Bearing, BearingFault
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
        raise NotImplementedError("不需要实例化,可以直接调用静态方法！")

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
        data = bearing.feature_data

        if bearing.stage_data is None or not is_staged:
            for key in bearing.feature_data:
                plt.plot(bearing.feature_data[key], label=key)
        else:
            fpt = bearing.stage_data.fpt_feature
            eol = bearing.stage_data.eol_feature
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
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)

        x = test_set.z.reshape(-1) / 60
        y = result.mean.reshape(-1)

        # 筛选出所有小于 1 的值
        filtered_array = test_set.y[test_set.y < 1]
        # 找到小于 1 的最大值
        max_value = filtered_array.max()
        # 找到该最大值在原数组中的下标
        max_index = np.where(test_set.y == max_value)[0][0]

        plt.plot([0, x[max_index], max(x)], [1, 1, 0], color='red')

        plt.scatter(x, y, label='Our proposed model', s=1)
        plt.title(f'RUL prediction result of {test_set.name}')
        plt.xlabel('Time (min)')
        plt.ylabel('Relative RUL')
        plt.legend()
        plt.show()

    @staticmethod
    def fault_during_time(test_set: Dataset, result: Result, interval=1):
        # todo 存在bug，当interval不能被行数整除时会报错
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)

        plt.ylim(-0.4, result.mean.shape[1] - 0.6)

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

        plt.scatter(x, y, label='Fault type', s=1)

        plt.title(f'Fault Type Prediction Result of {test_set.name}')
        plt.xlabel('Time (min)')
        plt.ylabel('Predicted Fault Label')
        plt.legend()
        plt.show()

    @staticmethod
    def fault_prediction_heatmap(test_set: Dataset, result: Result):
        """
        故障诊断热图（混淆矩阵图）单标签预测
        多标签预测无法使用，会出现不正常的数据 todo 待增加多标签预测的表示法
        :return:
        """
        plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)
        #  todo 没有考虑复合故障

        # 标签及标签数目
        labels = list(BearingFault.__members__)
        y_true = test_set.y
        # 当标签为类别索引时
        if y_true.shape[1] == 1:
            y_true = np.eye(len(labels))[y_true.squeeze().astype(int)]
        y_pred = result.mean

        # 找到每行最大值的索引
        max_indices = np.argmax(y_pred, axis=1)

        # 创建一个与原矩阵形状相同的全零矩阵
        result = np.zeros_like(y_pred)

        # 使用布尔索引将每行最大值的位置设为 1
        for i, idx in enumerate(max_indices):
            result[i, idx] = 1

        # 计算混淆矩阵
        conf_matrix = y_true.T @ result

        # 计算每一行的总和
        row_sums = conf_matrix.sum(axis=1, keepdims=True)

        # 将每个元素除以相应行的总和，并乘以 100
        conf_matrix_percent = np.zeros_like(conf_matrix, dtype=float)
        for i in range(len(labels)):
            if row_sums[i] != 0:
                conf_matrix_percent[i] = conf_matrix[i] / row_sums[i] * 100

        # conf_matrix_percent = conf_matrix_percent.astype(np.int).T

        # 绘制热图
        heatmap = sns.heatmap(conf_matrix_percent.T, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels,
                              yticklabels=labels, vmin=0, vmax=100)

        # 设置标签
        plt.xlabel('True label')
        plt.ylabel('Predicted label')

        # 将y轴文字恢复正常角度
        heatmap.set_yticklabels(labels, rotation=0)

        # 显示图形
        plt.show()

    @staticmethod
    def attention_heatmap(test_set: Dataset, result: Result):
        """
        生成注意力权重热图
        :return:
        """
        # 按时间排序
        sorted_indices = np.argsort(test_set.z.squeeze())
        data = result.mean[sorted_indices]

        # plt.figure(figsize=Plotter.__FIG_SIZE, dpi=Plotter.__DPI)
        # sns.heatmap(result.mean[:10, :10], annot=True, cmap='Reds', cbar=True)
        Plotter.show_heatmaps(data, 'Features', 'Inputs')
        # plt.xlabel('Inputs')
        # plt.ylabel('Features')
        # plt.title('Attention Weights')
        # plt.show()

    @staticmethod
    def show_heatmaps(matrices: ndarray, xlabel, ylabel):
        """显示矩阵热图"""
        # matrices = matrices.T
        matrices = np.expand_dims(matrices, axis=0)
        matrices = np.expand_dims(matrices, axis=0)
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=Plotter.__FIG_SIZE, sharex=True, sharey=True,
                                 squeeze=False)
        for i in range(num_rows):
            for j in range(num_cols):
                ax = axes[i, j]
                # sns.heatmap(matrices[i][j], ax=ax, cmap='Reds', cbar=True, vmin=0, vmax=1)
                sns.heatmap(matrices[i][j], ax=ax, cmap='Reds', cbar=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

                num_labels = matrices.shape[2]
                step = 100
                if num_labels > 1000:
                    step = 200
                elif num_labels > 2000:
                    step = 500
                ax.set_yticks(np.arange(0, num_labels, step))
                ax.set_yticklabels(np.arange(0, num_labels, step))

                num_labels = matrices.shape[3]
                step = 10  # 每隔多少个标签显示一次
                ax.set_xticks(np.arange(0, num_labels, step))
                ax.set_xticklabels(np.arange(0, num_labels, step))

                if i == num_rows - 1:
                    ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)

        plt.title('Attention Weights')
        plt.show()
