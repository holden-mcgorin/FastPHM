import os
from functools import wraps
import string
from typing import Union, List

from scipy.stats import mode

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from fastphm.data.Dataset import Dataset
from fastphm.entity.ABCEntity import ABCEntity
from fastphm.entity.Bearing import Bearing
from fastphm.model.ABCModel import ABCModel
from fastphm.model.Result import Result
from fastphm.system.Logger import Logger
from fastphm.util.ThresholdTrimmer import ThresholdTrimmer


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
        plt.tight_layout(pad=Plotter.PAD)
        plt.show()
        return title

    return wrapper


class Plotter:
    """
    画图器，所有的图片统一由画图器处理
    """

    # 画图设置
    DPI = 200  # 分辨率，默认100
    SIZE = (7, 4.8)  # 图片大小
    PAD = 0.5  # 图片边缘填充
    IS_LEGEND = True  # 是否输出图例
    FORMAT = 'svg'  # 图像输出格式：可选jpg, png, svg
    IS_SAVE = False  # 是否保存图像
    FIG_DIR = '.\\fig'  # 图像保存路径

    # 阶段划分颜色设置
    __COLOR_NORMAL_STAGE = 'green'
    __COLOR_DEGENERATION_STAGE = 'orange'
    __COLOR_FAILURE_STAGE = 'red'
    __COLOR_FAILURE_THRESHOLD = 'darkred'

    if not os.path.exists(FIG_DIR) and IS_SAVE:
        os.makedirs(FIG_DIR)

    @classmethod
    def reset(cls):
        cls.DPI = 200
        cls.SIZE = (7, 4.8)
        cls.PAD = 0.2

    @staticmethod
    @postprocess
    def loss(model: ABCModel):
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)
        plt.plot(range(0, len(model.loss)), model.loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

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
        if fpt is not None and eol is not None:
            plt.plot(np.arange(eol - fpt + 1) + fpt, data[fpt:eol + 1], label='degeneration stage',
                     color=Plotter.__COLOR_DEGENERATION_STAGE)
            plt.plot(np.arange(len(data[eol:])) + eol, data[eol:], label='failure stage',
                     color=Plotter.__COLOR_FAILURE_STAGE)
            # 绘制垂直线表示中间点
            plt.axvline(x=fpt, color='black', linestyle='--', label='EFP')
            plt.axvline(x=eol, color='skyblue', linestyle='--', label='EoL')
        if fpt is not None and eol is None:
            plt.plot(np.arange(len(data[fpt:])) + fpt, data[fpt:], label='degeneration stage',
                     color=Plotter.__COLOR_DEGENERATION_STAGE)
            plt.axvline(x=fpt, color='black', linestyle='--', label='EFP')

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
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        if entity.stage_data is None or not is_staged:
            for key in entity.raw_data.keys():
                y = entity.raw_data[key]
                x = np.arange(len(y))
                plt.plot(x, y, label=key)
                # plt.plot(x, y, label=key, color=Plotter.__COLOR_NORMAL_STAGE)
        else:
            fpt = entity.stage_data.fpt_raw
            eol = entity.stage_data.eol_raw
            data = entity.raw_data
            Plotter.__staged(data, fpt, eol)

        title = entity.name + ' Raw Sensor Signals'
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        if Plotter.IS_LEGEND:
            plt.legend()
        return title

    @staticmethod
    def __feature(entity: ABCEntity, is_staged=True):
        if entity.feature_data is None:
            Logger.warning('Entity object does not have feature data, drawing skipped')
            return
        data = entity.feature_data

        if entity.stage_data is None or not is_staged:
            for key in entity.feature_data:
                plt.plot(entity.feature_data[key], label=key)
        else:
            fpt = entity.stage_data.fpt_feature
            eol = entity.stage_data.eol_feature
            Plotter.__staged(data, fpt, eol)

            # 画失效阈值
            if entity.stage_data.failure_threshold_feature is not None:
                plt.axhline(y=entity.stage_data.failure_threshold_feature, color=Plotter.__COLOR_FAILURE_THRESHOLD,
                            label='failure threshold')

            # 添加文字标注 todo 这里默认特征值为一维的数据
            # ax = plt.gca()
            # x_lim = ax.get_xlim()
            # y_lim = ax.get_ylim()  # 获取 y_test 轴的上限和下限
            # if eol is not None:
            #     plt.text(eol + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]), 'EoL', color='black',
            #              fontsize=12)
            # if fpt is not None:
            #     plt.text(fpt + x_lim[1] / 75, y_lim[0] + 0.018 * (y_lim[1] - y_lim[0]), 'fpt', color='black',
            #              fontsize=12)

    @staticmethod
    @postprocess
    def feature(entity: ABCEntity, is_staged=True, label_x='Time (Sample Index)', label_y='feature value'):
        """
        绘画轴承特征图，当存在阶段数据且设为True时画阶段特征图
        :return:
        """
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        Plotter.__feature(entity, is_staged)

        title = entity.name + ' Feature Values'
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        if Plotter.IS_LEGEND:
            plt.legend()
        return title

    @staticmethod
    @postprocess
    def rul_degeneration(bearing: Bearing, result: Result, is_trim: bool = True, is_staged: bool = True):
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

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

        title = bearing.name + ' Degeneration Trend'
        plt.title(title)
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('processor value')
        if Plotter.IS_LEGEND:
            plt.legend()
        return title

    @staticmethod
    @postprocess
    def rul_end2end(test_set: Dataset, result: Union[Result, List[Result]],
                    is_scatter=True, label_x='Time', label_y='RUL'):
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        x = test_set.z.reshape(-1)
        results = []
        if isinstance(result, Result):
            results.append(result)
        else:
            results = result

        if is_scatter:
            for i, result in enumerate(results):
                y = result.outputs.reshape(-1)
                plt.scatter(x, y, label=result.name, s=2 * (len(results) - i))
        else:
            for result in results:
                y = result.outputs.reshape(-1)
                # 将数据按时间排序
                sorted_indices = np.argsort(x)
                # 重新排列矩阵的行
                x1 = x[sorted_indices]
                y = y[sorted_indices]
                plt.plot(x1, y, label=result.name)

        # 找到第二大的值在原数组中的下标（标准线的转折处）
        unique_array = np.unique(test_set.y)
        max_val = unique_array[-1]
        second_val = unique_array[-2]
        min_val = unique_array[0]
        max_index = np.where(test_set.y == second_val)[0][0]
        plt.axvline(x=x[max_index], color='black', linestyle='--', label='EFP')
        plt.plot([0, x[max_index], max(x)], [max_val, max_val, min_val], color='red', label='RUL target')

        title = 'RUL prediction result of ' + test_set.name
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        if Plotter.IS_LEGEND:
            plt.legend()
            # # 获取当前图中的句柄和标签
            # handles, labels = plt.gca().get_legend_handles_labels()
            # # 调整顺序
            # order = [1, 0, 2]  # 指定显示顺序为：标签2 -> 标签1
            # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        return title

    @staticmethod
    @postprocess
    def rul_end2end_batch(test_sets: [Dataset], results: [Result],
                          width: int, height: int,
                          is_scatter=True,
                          label_x='Time', label_y='RUL'):
        size = len(test_sets)
        new_figure_size = (Plotter.SIZE[0] * width / size * 4, Plotter.SIZE[1] * height / size * 4)
        plt.figure(figsize=new_figure_size, dpi=Plotter.DPI)

        for i in range(size):
            plt.subplot(height, width, i + 1)
            x = test_sets[i].z.reshape(-1)
            y = results[i].outputs.reshape(-1)

            # 找到第二大的值在原数组中的下标（画标准线）
            unique_array = np.unique(test_sets[i].y)
            max_val = unique_array[-1]
            second_val = unique_array[-2]
            max_index = np.where(test_sets[i].y == second_val)[0][0]
            x_efp = x[max_index]
            x_fpt = x[max_index]
            x_max = max(x)

            if is_scatter:
                plt.scatter(x, y, label='RUL Predicted', s=3)
            else:
                # 将数据按时间排序
                sorted_indices = np.argsort(x)
                # 重新排列矩阵的行
                x = x[sorted_indices]
                y = y[sorted_indices]
                plt.plot(x, y, marker='o', markersize=2, label='RUL Predicted')

            plt.axvline(x=x_efp, color='black', linestyle='--', label='EFP')
            plt.plot([0, x_fpt, x_max], [max_val, max_val, 0], linestyle='-', color='red', label='RUL target')

            plt.title(f"({string.ascii_lowercase[i]}) {test_sets[i].name}")
            plt.xlabel(label_x)
            plt.ylabel(label_y)
            if Plotter.IS_LEGEND:
                plt.legend()

        # plt.tight_layout(pad=2.0)
        plt.tight_layout()

        return None

    @staticmethod
    @postprocess
    def rul_ascending(test_set: Dataset, result: Union[Result, List[Result]],
                      is_scatter=True, label_x='Time', label_y='RUL'):
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        # 获取RUL标签升序序列
        y = test_set.y.reshape(-1)
        y_hat = result.outputs.reshape(-1)
        sort_indices = np.argsort(y)
        y = y[sort_indices]
        y_hat = y_hat[sort_indices]

        # 画标签数据
        plt.plot(np.arange(1, y.shape[0] + 1), y, color='red', label='RUL target')
        # 画预测数据
        if is_scatter:
            plt.scatter(np.arange(1, y.shape[0] + 1), y_hat, label=result.name, s=2)
        else:
            plt.plot(np.arange(1, y.shape[0] + 1), y_hat, label=result.name, marker='o', markersize=4)

        title = 'RUL prediction result of ' + test_set.name
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        if Plotter.IS_LEGEND:
            plt.legend()
        return title

    @staticmethod
    @postprocess
    def fault_evolution(test_set: Dataset, result: Result, types: list):
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)
        plt.ylim(-0.4, result.outputs.shape[1] - 0.6)

        y_pred = (result.outputs > 0.5).astype(int)

        # 获取类别数（即列数）
        n_categories = len(types)

        # 为了刻度从0开始
        plt.scatter(0, 0, color='white')

        for category in range(n_categories):
            # 获取当前类别值为1的行索引
            indices = np.where(y_pred[:, category] == 1)[0]
            x = test_set.z.reshape(-1)[indices]
            plt.scatter(x, category + x * 0, s=1)

        # 设置 y 轴标签
        plt.yticks(ticks=np.arange(len(types)), labels=types)
        plt.xlabel('Time (min)')
        plt.ylabel('Predicted Fault Type')
        # if Plotter.IS_LEGEND:
        #     plt.legend()

    @staticmethod
    @postprocess
    def fault_evolution_batch(test_sets: [Dataset], results: [Result],
                              width: int, height: int,
                              types: list):
        size = len(test_sets)
        new_figure_size = (Plotter.SIZE[0] * width / size * 4, Plotter.SIZE[1] * height / size * 4)
        plt.figure(figsize=new_figure_size, dpi=Plotter.DPI)

        for i in range(size):
            plt.subplot(height, width, i + 1)
            plt.ylim(-0.4, results[i].outputs.shape[1] - 0.6)

            y_pred = (results[i].outputs > 0.5).astype(int)

            # 获取类别数（即列数）
            n_categories = len(types)
            for category in range(n_categories):
                # 获取当前类别值为1的行索引
                indices = np.where(y_pred[:, category] == 1)[0]
                x = test_sets[i].z.reshape(-1)[indices]
                plt.scatter(x, n_categories - category - 1 + x * 0, s=1, label=types[n_categories - category - 1])

            # 为了刻度从0开始
            plt.scatter(0, 0, color='white')

            plt.title(f"({string.ascii_lowercase[i]}) {test_sets[i].name}")
            plt.yticks(ticks=np.arange(len(types)), labels=types[::-1])
            plt.xlabel('Time (min)')
            plt.ylabel('Predicted Fault Type')
            # if Plotter.IS_LEGEND:
            #     plt.legend()

        plt.tight_layout()

        return None

    @staticmethod
    @postprocess
    def fault_diagnosis_evolution(test_set: Dataset, result: Result, types: list, interval=1):
        # todo 存在bug，当interval不能被行数整除时会报错
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        plt.ylim(-0.4, result.outputs.shape[1] - 0.6)

        x = test_set.z
        y = np.argmax(result.outputs, axis=1).reshape(-1, 1)  # 找出每行最大值的下标

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
        if Plotter.IS_LEGEND:
            plt.legend()

        return title

    @staticmethod
    @postprocess
    def fault_diagnosis_evolution_batch(test_sets: [Dataset], results: [Result],
                                        width: int, height: int,
                                        types: list):
        # todo 存在bug，当interval不能被行数整除时会报错
        size = len(test_sets)
        new_figure_size = (Plotter.SIZE[0] * width / size * 4, Plotter.SIZE[1] * height / size * 4)
        plt.figure(figsize=new_figure_size, dpi=Plotter.DPI)

        for i in range(size):
            plt.subplot(height, width, i + 1)
            plt.ylim(-0.4, results[i].outputs.shape[1] - 0.6)

            x = test_sets[i].z
            y = np.argmax(results[i].outputs, axis=1)  # 找出每行最大值的下标
            y = y.reshape(-1, 1)

            # 将数据按时间排序
            xy = np.hstack((x, y))
            last_column_index = xy[:, 0]
            sorted_indices = np.argsort(last_column_index)
            # 重新排列矩阵的行
            xy = xy[sorted_indices]

            x = xy[:, 0]
            y = xy[:, 1]

            plt.scatter(x, y, label='Fault type', s=1)

            # 设置 y 轴标签
            plt.yticks(ticks=np.arange(len(types)), labels=types)

            plt.title(f"({string.ascii_lowercase[i]}) {test_sets[i].name}")
            plt.xlabel('Time (min)')
            plt.ylabel('Predicted Fault Type')
            if Plotter.IS_LEGEND:
                plt.legend()

        return None

    @staticmethod
    @postprocess
    def diagnosis_confusion_matrix_compound(test_set: Dataset, result: Result, types: list,
                                            width: int, height: int):
        """
        故障诊断混淆矩阵图（复合故障）（多标签分类）
        """
        from sklearn.metrics import multilabel_confusion_matrix

        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        prediction = np.where(result.outputs > 0.5, 1, 0)
        mcm = multilabel_confusion_matrix(test_set.y, prediction)

        fig, axes = plt.subplots(height, width)
        if width == 1 or height == 1:  # 如果只有一个标签
            axes = np.expand_dims(axes, axis=0)

        for i in range(len(types)):
            ax = axes[i // width, i % width]  # 获取具体的子图

            # 计算每个标签的混淆矩阵的百分比
            conf_matrix_percent = mcm[i] / mcm[i].sum(axis=1)[:, np.newaxis]

            # 使用 seaborn 画混淆矩阵热图，自动调整字体颜色
            heatmap = sns.heatmap(conf_matrix_percent, annot=True, fmt=".2%", cmap="Blues",
                                  xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], vmin=0,
                                  vmax=1,
                                  cbar=False, ax=ax, square=True)

            ax.set_title(f"({string.ascii_lowercase[i]}) {types[i].name}")
            ax.set_xlabel('True label')
            ax.set_ylabel('Predicted label')

    @staticmethod
    @postprocess
    def diagnosis_confusion_matrix(test_set: Dataset, result: Result, types: list):
        """
        故障诊断热图（混淆矩阵图）单标签预测（多分类）
        多标签预测无法使用，会出现不正常的数据 todo 待增加多标签预测的表示法
        :return:
        """
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

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
        # plt.title(title)

        plt.xlabel('True label')
        plt.ylabel('Predicted label')

        return title

    @staticmethod
    @postprocess
    def attention_heatmap_batch(test_sets: [Dataset], results: [Result],
                                height: int, width: int,
                                label_x='Inputs', label_y='Features'):

        size = len(test_sets)
        new_figure_size = (Plotter.SIZE[0] * width / size * 4, Plotter.SIZE[1] * height / size * 4)

        fig, axes = plt.subplots(height, width, figsize=new_figure_size, dpi=Plotter.DPI, sharex=False, sharey=False,
                                 squeeze=False)

        for i in range(height):
            for j in range(width):
                index = width * i + j
                if size <= index:
                    continue

                sorted_indices = np.argsort(test_sets[index].z.squeeze())
                data = results[index].outputs[sorted_indices].T

                ax = axes[i, j]
                # sns.heatmap(matrices[i][j], ax=ax, cmap='Reds', cbar=True, vmin=0, vmax=1)
                sns.heatmap(data, ax=ax, cmap='viridis', cbar=True)
                # sns.heatmap(data, ax=ax, cmap='Reds', cbar=True)

                # 设置y轴
                num_y = data.shape[0]  # 注意力特征数/专家数
                step = 1
                yticks = np.arange(0, num_y, step)
                ax.set_yticks(yticks + 0.5)
                ax.set_yticklabels(yticks, rotation=270)

                # 设置x轴
                num_x = data.shape[1]
                step = 1
                if num_x > 10:
                    step = num_x // 10
                if num_x > 100:
                    step = 50
                if num_x > 200:
                    step = 50
                if num_x > 500:
                    step = 100
                if num_x > 1000:
                    step = 200
                if num_x > 1500:
                    step = 300
                if num_x > 2000:
                    step = 400
                if num_x > 2500:
                    step = 600
                if num_x > 5000:
                    step = 1500
                if num_x > 10000:
                    step = 2500
                yticks = np.arange(0, num_x, step)
                ax.set_xticks(yticks)
                ax.set_xticklabels(yticks, rotation=0, ha='center')

                title = test_sets[index].name
                ax.set_title(f"({string.ascii_lowercase[index]}) {title}")

                ax.set_xlabel(label_x)
                ax.set_ylabel(label_y)

        plt.tight_layout()
        return None

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

        # matrices = matrices.T
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=0)
        num_rows, num_cols = data.shape[0], data.shape[1]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=Plotter.SIZE, sharex=True, sharey=True,
                                 squeeze=False)
        for i in range(num_rows):
            for j in range(num_cols):
                ax = axes[i, j]
                # sns.heatmap(matrices[i][j], ax=ax, cmap='Reds', cbar=True, vmin=0, vmax=1)
                sns.heatmap(data[i][j], ax=ax, cmap='Reds', cbar=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

                num_labels = data.shape[2]  # 测试样本数

                # 设置刻度
                step = num_labels // 10
                # step = round(step / 100) * 100
                # if step == 0:
                #     step = 1

                yticks = np.arange(0, num_labels, step)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)

                num_labels = data.shape[3]  # 注意力特征数

                step = 1
                if num_labels > 10:
                    step = num_labels // 10
                elif num_labels > 100:
                    step = 100
                elif num_labels > 1000:
                    step = 200
                elif num_labels > 2000:
                    step = 500
                ax.set_xticks(np.arange(0, num_labels, step) + 0.5)
                ax.set_xticklabels(np.arange(0, num_labels, step), rotation=0, ha='center')

                if i == num_rows - 1:
                    ax.set_xlabel('Features')
                if j == 0:
                    ax.set_ylabel('Inputs')

        title = 'Attention Weights of ' + test_set.name
        plt.title(title)
        return title

    @staticmethod
    @postprocess
    def fault_diagnosis_feature(test_set: Dataset, result: Result, types: list, title=None):
        """
        针对不同的故障类型，使用T-SNE降维,使特征可视化
        :param title:
        :param types: 故障类型
        :param test_set: 故障诊断测试集
        :param result: 高维特征
        :return:
        """

        # 生成二维特征
        from fastphm.util.T_SNE import T_SNE
        features = result.outputs
        features_t = T_SNE.fit(features)
        y = test_set.y.reshape(-1)

        # 绘制散点图
        for i in range(len(types)):
            plt.scatter(features_t[y == i, 0], features_t[y == i, 1], label=types[i])

        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        if title is not None:
            plt.title(title)
