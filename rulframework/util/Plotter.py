import os
from functools import wraps

from numpy import ndarray
from scipy.stats import mode

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from rulframework.data.Dataset import Dataset
from rulframework.entity.ABCEntity import ABCEntity
from rulframework.entity.Bearing import Bearing, Fault
from rulframework.model.ABCModel import ABCModel
from rulframework.model.Result import Result
from rulframework.util.ThresholdTrimmer import ThresholdTrimmer


def postprocess(func):
    """
    所有画图方法的后置处理
    1. 是否保存图片
    :param func:
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        title = func(*args, **kwargs)
        if Plotter.IS_SAVE:
            plt.savefig(os.path.join(Plotter.FIG_DIR, title + '.' + Plotter.FORMAT), format=Plotter.FORMAT)
        plt.show()
        return title

    return wrapper


class Plotter:
    """
    画图器，所有的图片统一由画图器处理
    """

    # 画图设置
    __DPI = 200  # 分辨率，默认100
    __SIZE = (10, 6)  # 图片大小
    __COLOR_NORMAL_STAGE = 'green'
    __COLOR_DEGENERATION_STAGE = 'orange'
    __COLOR_FAILURE_STAGE = 'red'
    __COLOR_FAILURE_THRESHOLD = 'darkred'

    # 图片保存设置
    IS_SAVE = False
    # IS_SAVE = False
    FORMAT = 'svg'
    # FORMAT = 'jpg'
    # FORMAT = 'png'
    FIG_DIR = '.\\fig'

    if not os.path.exists(FIG_DIR) and IS_SAVE:
        os.makedirs(FIG_DIR)

    def __init__(self):
        raise NotImplementedError("不需要实例化,可以直接调用静态方法！")

    @classmethod
    def set_dpi(cls, dpi: int):
        cls.__DPI = dpi

    @classmethod
    def set_size(cls, size: (int, int)):
        cls.__SIZE = size

    @staticmethod
    @postprocess
    def loss(model: ABCModel):
        plt.plot(range(0, len(model.loss)), model.loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        title = 'Training Loss Over Epochs'
        plt.title(title)
        plt.legend()
        return title

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
    @postprocess
    def raw(entity: ABCEntity, is_staged=True, label_x='Time (Sample Index)', label_y='value'):
        """
        绘画原始振动信号图像
        :param label_y:
        :param label_x:
        :param entity:需要画图的对象
        :param is_staged:是否划分轴承退化阶段
        :return:
        """
        plt.figure(figsize=Plotter.__SIZE, dpi=Plotter.__DPI)

        if entity.stage_data is None or not is_staged:
            for key in entity.raw_data.keys():
                y = entity.raw_data[key]
                x = np.arange(len(y))
                plt.plot(x, y, label=key)
        else:
            fpt = entity.stage_data.fpt_raw
            eol = entity.stage_data.eol_raw
            data = entity.raw_data
            Plotter.__staged(data, fpt, eol)

        title = entity.name + ' Raw Sensor Signals'
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend()
        return title

    @staticmethod
    def __feature(entity: ABCEntity, is_staged=True):
        data = entity.feature_data

        if entity.stage_data is None or not is_staged:
            for key in entity.feature_data:
                plt.plot(entity.feature_data[key], label=key)
        else:
            fpt = entity.stage_data.fpt_feature
            eol = entity.stage_data.eol_feature
            Plotter.__staged(data, fpt, eol)
            # 画失效阈值
            plt.axhline(y=entity.stage_data.failure_threshold_feature, color=Plotter.__COLOR_FAILURE_THRESHOLD,
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
    @postprocess
    def feature(entity: ABCEntity, is_staged=True, label_x='Time (Sample Index)', label_y='feature value'):
        """
        绘画轴承特征图，当存在阶段数据且设为True时画阶段特征图
        :return:
        """
        plt.figure(figsize=Plotter.__SIZE, dpi=Plotter.__DPI)

        Plotter.__feature(entity, is_staged)

        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.gca().add_artist(legend)
        title = entity.name + ' Feature Values'
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        return title

    @staticmethod
    @postprocess
    def rul_degeneration(bearing: Bearing, result: Result, is_trim: bool = True, is_staged: bool = True):
        plt.figure(figsize=Plotter.__SIZE, dpi=Plotter.__DPI)

        Plotter.__feature(bearing, is_staged)

        """
        画退化曲线
        """
        if result.mean is None:
            result.mean = result.outputs.squeeze()

        if is_trim:
            trimmer = ThresholdTrimmer(bearing.stage_data.failure_threshold_feature)
            result = trimmer.trim(result)

        # 画预测值（确定性预测和不确定性预测）
        x = np.arange(result.mean.shape[0] + 1) + result.begin_index
        y = np.hstack((np.array([bearing.feature_data.values[result.begin_index, 0]]), result.mean))
        plt.plot(x, y, label='mean')

        # 画置信区间（不确定性预测）
        if result.lower is not None and result.upper is not None:
            x = np.arange(len(result.lower) + 1) + result.begin_index
            lower = np.hstack((bearing.feature_data.values[result.begin_index, 0], result.lower))
            upper = np.hstack((bearing.feature_data.values[result.begin_index, 0], result.upper))
            plt.fill_between(x, lower, upper, alpha=0.25, label='confidence interval')

        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.gca().add_artist(legend)
        title = bearing.name + ' Degeneration Trend'
        plt.title(title)
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('processor value')
        return title

    @staticmethod
    @postprocess
    def rul_end2end(test_set: Dataset, result: Result, is_scatter=True, label_x='Time', label_y='RUL'):
        plt.figure(figsize=Plotter.__SIZE, dpi=Plotter.__DPI)

        x = test_set.z.reshape(-1)
        y = result.outputs.reshape(-1)

        # 找到第二大的值在原数组中的下标（画标准线）
        unique_array = np.unique(test_set.y)
        max_val = unique_array[-1]
        second_val = unique_array[-2]
        max_index = np.where(test_set.y == second_val)[0][0]
        plt.plot([0, x[max_index], max(x)], [max_val, max_val, 0], color='red')

        if is_scatter:
            plt.scatter(x, y, label='Our proposed model', s=1)
        else:
            # 将数据按时间排序
            sorted_indices = np.argsort(x)
            # 重新排列矩阵的行
            x = x[sorted_indices]
            y = y[sorted_indices]
            plt.plot(x, y, label='Our proposed model')

        title = 'RUL prediction result of ' + test_set.name
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend()
        return title

    @staticmethod
    @postprocess
    def fault_diagnosis_evolution(test_set: Dataset, result: Result, types: list, interval=1):
        # todo 存在bug，当interval不能被行数整除时会报错
        plt.figure(figsize=Plotter.__SIZE, dpi=Plotter.__DPI)

        plt.ylim(-0.4, result.outputs.shape[1] - 0.6)

        x = test_set.z / 60
        y = np.argmax(result.outputs, axis=1)  # 找出每行最大值的下标
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
        y = np.apply_along_axis(lambda l: mode(l, keepdims=False)[0], axis=1, arr=y).reshape(-1)

        plt.scatter(x, y, label='Fault type', s=1)

        # 设置 y 轴标签
        plt.yticks(ticks=np.arange(len(types)), labels=types)

        title = 'Fault Type Prediction Result of ' + test_set.name
        plt.title(title)
        plt.xlabel('Time (min)')
        plt.ylabel('Predicted Fault Type')
        plt.legend()
        return title

    @staticmethod
    @postprocess
    def fault_diagnosis_heatmap(test_set: Dataset, result: Result, types: list):
        """
        故障诊断热图（混淆矩阵图）单标签预测
        多标签预测无法使用，会出现不正常的数据 todo 待增加多标签预测的表示法
        :return:
        """
        plt.figure(figsize=Plotter.__SIZE, dpi=Plotter.__DPI)
        #  todo 没有考虑复合故障

        # 标签及标签数目
        labels = list(types)
        y_true = test_set.y
        # 当标签为类别索引时
        if y_true.shape[1] == 1:
            y_true = np.eye(len(labels))[y_true.squeeze().astype(int)]
        y_pred = result.outputs

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
                conf_matrix_percent[i] = conf_matrix[i] / row_sums[i]

        # conf_matrix_percent = conf_matrix_percent.astype(np.int).T

        # 绘制热图
        heatmap = sns.heatmap(conf_matrix_percent.T, annot=True, fmt=".2%", cmap="Blues", xticklabels=labels,
                              yticklabels=labels, vmin=0, vmax=1)
        # 将y轴文字恢复正常角度
        heatmap.set_yticklabels(labels, rotation=0)

        # 设置标签
        title = 'Accuracy of Fault Diagnosis'
        plt.title(title)
        plt.xlabel('True labeler')
        plt.ylabel('Predicted labeler')

        return title

    @staticmethod
    @postprocess
    def attention_heatmap(test_set: Dataset, result: Result):
        """
        生成注意力权重热图
        :return:
        """
        # 按时间排序
        sorted_indices = np.argsort(test_set.z.squeeze())
        data = result.outputs[sorted_indices]
        Plotter.show_heatmaps(data, 'Features', 'Inputs')
        title = 'Attention Weights of ' + test_set.name
        plt.title(title)
        return title

    @staticmethod
    def show_heatmaps(matrices: ndarray, x_label, y_label):
        """显示矩阵热图"""
        # matrices = matrices.T
        matrices = np.expand_dims(matrices, axis=0)
        matrices = np.expand_dims(matrices, axis=0)
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=Plotter.__SIZE, sharex=True, sharey=True,
                                 squeeze=False)
        for i in range(num_rows):
            for j in range(num_cols):
                ax = axes[i, j]
                # sns.heatmap(matrices[i][j], ax=ax, cmap='Reds', cbar=True, vmin=0, vmax=1)
                sns.heatmap(matrices[i][j], ax=ax, cmap='Reds', cbar=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

                num_labels = matrices.shape[2]  # 测试样本数
                # 计算步长，使得总共显示10个刻度
                step = num_labels // 10
                # 计算适合的步长，使刻度均匀分布并去掉零头
                # 使步长为最接近 step 的最小100的倍数
                step = round(step / 100) * 100
                if step == 0:
                    step = 1

                # 设置刻度
                yticks = np.arange(0, num_labels, step)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)

                num_labels = matrices.shape[3]  # 注意力特征数
                step = num_labels // 10
                if num_labels > 100:
                    step = 100
                elif num_labels > 1000:
                    step = 200
                elif num_labels > 2000:
                    step = 500
                ax.set_xticks(np.arange(0, num_labels, step) + 0.5)
                ax.set_xticklabels(np.arange(0, num_labels, step), rotation=0, ha='center')

                if i == num_rows - 1:
                    ax.set_xlabel(x_label)
                if j == 0:
                    ax.set_ylabel(y_label)
