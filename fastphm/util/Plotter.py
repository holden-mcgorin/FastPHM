import math
import os
from functools import wraps
import string
from typing import Union, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from fastphm.data.Dataset import Dataset
from fastphm.entity.ABCEntity import ABCEntity
from fastphm.entity.Bearing import Bearing
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
        if Plotter.IS_LEGEND:
            # 获取当前图中的所有图形元素
            handles, labels = plt.gca().get_legend_handles_labels()
            # 过滤掉被 matplotlib 忽略的 label（以 _ 开头的）
            filtered = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_')]
            # 如果存在有效的图例项，则添加 legend
            if filtered:
                plt.legend()
        plt.tight_layout(pad=Plotter.PAD)
        plt.show()
        return title

    return wrapper


class Plotter:
    """
    画图器，所有的图片统一由画图器处理
    """

    # 画图设置(注意：DPI或SIZE过大会报错)
    DPI = 200  # 分辨率，默认100
    SIZE = (5, 3.5)  # 图片大小
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
    def loss(loss_history_dicts: dict):
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)
        for key in loss_history_dicts.keys():
            plt.plot(loss_history_dicts[key], label=key)
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
        plt.plot(np.arange(fpt + 1), data[:fpt + 1], label='Normal stage', color=Plotter.__COLOR_NORMAL_STAGE)
        if fpt is not None and eol is not None:
            plt.plot(np.arange(eol - fpt + 1) + fpt, data[fpt:eol + 1], label='Degeneration stage',
                     color=Plotter.__COLOR_DEGENERATION_STAGE)
            plt.plot(np.arange(len(data[eol:])) + eol, data[eol:], label='Failure stage',
                     color=Plotter.__COLOR_FAILURE_STAGE)
            # 绘制垂直线表示中间点
            plt.axvline(x=fpt, color='black', linestyle='--', label='EFP')
            plt.axvline(x=eol, color='skyblue', linestyle='--', label='EoL')
        if fpt is not None and eol is None:
            plt.plot(np.arange(len(data[fpt:])) + fpt, data[fpt:], label='Degeneration stage',
                     color=Plotter.__COLOR_DEGENERATION_STAGE)
            plt.axvline(x=fpt, color='black', linestyle='--', label='Change point')

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
    @postprocess
    def raw3d(entity: ABCEntity):
        fig = plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)
        ax = fig.add_subplot(111, projection='3d')

        for sensor_idx, key in enumerate(entity.raw_data.keys()):
            z = entity.raw_data[key]  # 当前传感器的值
            x = np.arange(len(z))  # 时间
            y = np.full_like(x, sensor_idx)  # 将每个传感器放在不同的z平面上
            ax.plot(x, y, z, label=f'{key}')

        title = entity.name + ' Raw Sensor Signals'
        # 设置标签
        ax.set_xlabel('Time')
        ax.set_ylabel('Index')
        ax.set_zlabel('Value')
        ax.set_title(title)

        if Plotter.IS_LEGEND:
            plt.legend()
        return title

    @staticmethod
    @postprocess
    def feature3d(entity: ABCEntity):
        fig = plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)
        ax = fig.add_subplot(111, projection='3d')

        for sensor_idx, key in enumerate(entity.feature_data.keys()):
            z = entity.feature_data[key]  # 当前传感器的值
            x = np.arange(len(z))  # 时间
            y = np.full_like(x, sensor_idx)  # 将每个传感器放在不同的z平面上
            ax.plot(x, y, z, label=f'{key}')

        title = entity.name + ' Raw Sensor Signals'
        # 设置标签
        ax.set_xlabel('Time')
        ax.set_ylabel('Index')
        ax.set_zlabel('Value')
        ax.set_title(title)

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
            result.mean = result.y_hat.squeeze()

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
    def rul_end2end(test_set: Dataset, result: Result,
                    is_scatter=True, cols=None, rows=None,
                    label_x='Time', label_y='RUL', title='default'):

        num_fig = len(test_set.entity_map)
        cols, rows = Plotter.__auto_cols_rows(num_fig, cols, rows)

        plt.figure(figsize=(Plotter.SIZE[0] * cols, Plotter.SIZE[1] * rows), dpi=Plotter.DPI)

        test_sets = test_set.split_by_entity()
        results = result.split_by_entity()

        for i in range(num_fig):
            plt.subplot(rows, cols, i + 1)
            x = test_sets[i].z.reshape(-1)
            y = results[i].y_hat.reshape(-1)

            # 画辅助线
            unique_array = np.unique(test_sets[i].y.reshape(-1))
            max_y = unique_array[-1]
            min_y = unique_array[0]
            max_x = max(x)
            min_x = min(x)
            if np.where(test_sets[i].y == max_y)[0].shape[0] > 2 and unique_array.shape[0] > 2:  # 存在FPT
                fpt_y = unique_array[-2]
                fpt_index = np.where(test_sets[i].y == fpt_y)[0][0]
                fpt_x = x[fpt_index]
                # 画标准线和fpt线
                plt.axvline(x=fpt_x, color='black', linestyle='--', label='FPT')
                plt.plot([min_x, fpt_x, max_x], [max_y, max_y, min_y], linestyle='-', color='red', label='RUL target')
            else:  # 不存在FPT
                plt.plot([min_x, max_x], [max_y, min_y], linestyle='-', color='red', label='RUL target')

            # 画预测线
            if is_scatter:
                plt.scatter(x, y, label='RUL Predicted', s=3)
            else:
                # 将数据按时间排序
                sorted_indices = np.argsort(x)
                # 重新排列矩阵的行
                x = x[sorted_indices]
                y = y[sorted_indices]
                plt.plot(x, y, marker='o', markersize=2, label='RUL Predicted')

            sub_title = test_sets[i].name if title == 'default' else title
            if num_fig == 1:
                plt.title(sub_title)
            elif num_fig <= 26:
                plt.title(f"({string.ascii_lowercase[i]}) {sub_title}")
            else:
                plt.title(f"({i + 1}) {sub_title}")

            plt.xlabel(label_x)
            plt.ylabel(label_y)
            if Plotter.IS_LEGEND:
                plt.legend()

        if Plotter.IS_LEGEND:
            plt.legend()
        return title

    @staticmethod
    @postprocess
    def rul_ascending(test_set: Dataset, result: Union[Result, List[Result]],
                      is_scatter=True, label_x='Time', label_y='RUL', title='RUL prediction result'):
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        # 获取RUL标签升序序列
        y = test_set.y.reshape(-1)
        y_hat = result.y_hat.reshape(-1)
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

        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(title)

        if Plotter.IS_LEGEND:
            plt.legend()
        return title

    @staticmethod
    @postprocess
    def fault_evolution(test_set: Dataset, result: Result, types: list,
                        cols=None, rows=None, title='default'):
        """
        故障演化图
        """
        num_fig = len(test_set.entity_map)
        cols, rows = Plotter.__auto_cols_rows(num_fig, cols, rows)

        plt.figure(figsize=(Plotter.SIZE[0] * cols, Plotter.SIZE[1] * rows), dpi=Plotter.DPI)

        test_sets = test_set.split_by_entity()
        results = result.split_by_entity()

        for i in range(num_fig):
            plt.subplot(rows, cols, i + 1)

            # 为了刻度从0开始
            plt.scatter(0, 0, color='white')
            plt.ylim(-0.4, results[i].y_hat.shape[1] - 0.6)

            # 判断任务类型是多分类还是多标签分类
            task_type = 'unknown'
            if test_sets[i].y.shape[1] == 1:
                task_type = 'multi class'
            if np.any((results[i].y_hat < 0) | (results[i].y_hat > 1)):
                task_type = 'multi class'

            # 离散化预测值
            if task_type == 'multi class':
                y_pred = np.zeros_like(results[i].y_hat)
                max_indices = np.argmax(results[i].y_hat, axis=1)
                y_pred[np.arange(results[i].y_hat.shape[0]), max_indices] = 1
            else:  # 'multi label'
                y_pred = (results[i].y_hat > 0.5).astype(int)

            # 画预测结果
            for category in range(len(types)):
                # 获取当前类别值为1的行索引
                indices = np.where(y_pred[:, category] == 1)[0]
                x = test_sets[i].z.reshape(-1)[indices]
                plt.scatter(x, category + x * 0, s=1)

            # 画早期故障点
            sort = np.argsort(test_sets[i].z.reshape(-1))
            sort_y = test_sets[i].y[sort]
            sort_z = test_sets[i].z[sort]
            mask = np.any(sort_y != 0, axis=1)  # 如果该行中有至少一个元素非0，则为 True
            first_nonzero_index = np.argmax(mask)
            z_efp = sort_z[first_nonzero_index]
            if first_nonzero_index != 0:  # 没有健康阶段则不画FPT线
                plt.axvline(x=z_efp, color='black', linestyle='--', label='EFP')

            # 设置 y 轴标签
            plt.yticks(ticks=np.arange(len(types)), labels=types)
            plt.xlabel('Time (min)')
            plt.ylabel('Predicted Fault Type')

            sub_title = test_sets[i].name if title == 'default' else title
            if num_fig == 1:
                plt.title(sub_title)
            elif num_fig <= 26:
                plt.title(f"({string.ascii_lowercase[i]}) {sub_title}")
            else:
                plt.title(f"({i + 1}) {sub_title}")

            # if Plotter.IS_LEGEND:
            #     plt.legend()
        plt.tight_layout()

    @staticmethod
    def confusion_matrix(test_set: Dataset, result: Result, types: list):
        # 判断任务类型是多分类还是多标签分类
        task_type = 'unknown'
        if test_set.y.shape[1] == 1:
            task_type = 'multi class'
        if np.any((result.y_hat < 0) | (result.y_hat > 1)):
            task_type = 'multi class'

        if task_type == 'multi class':
            Plotter.confusion_matrix_multiclass(test_set=test_set, result=result, types=types)
        else:
            Plotter.confusion_matrix_multilabel(test_set=test_set, result=result, types=types)

    @staticmethod
    def __auto_cols_rows(num_fig, cols, rows) -> (int, int):
        if not (cols and rows):
            cols = math.ceil(math.sqrt(num_fig))
            rows = math.ceil(num_fig / cols)
        return cols, rows

    @staticmethod
    @postprocess
    def confusion_matrix_multilabel(test_set: Dataset, result: Result, types: list,
                                    cols: int = None, rows: int = None):
        """
        故障诊断混淆矩阵图（复合故障）（多标签分类）
        """
        num_fig = len(types)
        cols, rows = Plotter.__auto_cols_rows(num_fig, cols, rows)

        from sklearn.metrics import multilabel_confusion_matrix

        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        # 阈值二值化
        y_true = (test_set.y > 0.5).astype(int)
        y_pred = (result.y_hat > 0.5).astype(int)
        mcm = multilabel_confusion_matrix(y_true, y_pred)

        fig, axes = plt.subplots(rows, cols)
        if cols == 1 or rows == 1:  # 如果只有一个标签
            axes = np.expand_dims(axes, axis=0)

        for i in range(num_fig):
            ax = axes[i // cols, i % cols]  # 获取具体的子图

            # 计算每个标签的混淆矩阵的百分比
            row_sums = mcm[i].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 防止除以 0
            conf_matrix_percent = mcm[i] / row_sums

            # 使用 seaborn 画混淆矩阵热图，自动调整字体颜色
            sns.heatmap(conf_matrix_percent, annot=True, fmt=".2%", cmap="Blues",
                        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                        vmin=0, vmax=1, cbar=False, ax=ax, square=True)

            ax.set_title(f"({string.ascii_lowercase[i]}) {types[i].name}")
            ax.set_xlabel('True label')
            ax.set_ylabel('Predicted label')

    @staticmethod
    @postprocess
    def confusion_matrix_multiclass(test_set: Dataset, result: Result, types: list):
        """
        故障诊断热图（混淆矩阵图）单标签预测（多分类）
        多标签预测无法使用，会出现不正常的数据（只适用多分类而不是多标签分类）
        :return:
        """
        plt.figure(figsize=Plotter.SIZE, dpi=Plotter.DPI)

        # 标签及标签数目
        labels = list(types)
        y_true = test_set.y
        # 当标签为类别索引时
        if y_true.shape[1] == 1:
            y_true = np.eye(len(labels))[y_true.squeeze().astype(int)]
        y_pred = result.y_hat

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
    def attention(test_set: Dataset, result: Result,
                  cols: int = None, rows: int = None,
                  label_x='Inputs', label_y='Features'):

        num_fig = len(test_set.entity_map)
        cols, rows = Plotter.__auto_cols_rows(num_fig, cols, rows)

        test_sets = test_set.split_by_entity()
        results = result.split_by_entity()

        fig, axes = plt.subplots(rows, cols, squeeze=False,
                                 figsize=(Plotter.SIZE[0] * cols, Plotter.SIZE[1] * rows),
                                 dpi=Plotter.DPI)

        for i in range(rows):
            for j in range(cols):
                index = cols * i + j
                if num_fig <= index:
                    continue

                sorted_indices = np.argsort(test_sets[index].z.squeeze())
                data = results[index].y_hat[sorted_indices].T

                ax = axes[i, j]
                # sns.heatmap(matrices[i][j], ax=ax, cmap='Reds', cbar=True, vmin=0, vmax=1)
                # sns.heatmap(data, ax=ax, cmap='viridis', cbar=True)
                sns.heatmap(data, ax=ax, cmap='Reds', cbar=True)

                # 设置y轴
                num_y = data.shape[0]  # 注意力特征数/专家数
                step = 1
                yticks = np.arange(0, num_y, step)
                ax.set_yticks(yticks + 0.5)
                ax.set_yticklabels(yticks, rotation=270)

                num_x = data.shape[1]
                num_ticks = 6  # 你想要的总刻度数（包括起点和终点）

                if num_x <= num_ticks:
                    xticks = np.arange(num_x)
                else:
                    xticks = np.linspace(0, num_x - 1, num_ticks, dtype=int)

                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks, rotation=0, ha='center')

                title = test_sets[index].name
                ax.set_title(f"({string.ascii_lowercase[index]}) {title}")

                ax.set_xlabel(label_x)
                ax.set_ylabel(label_y)

        plt.tight_layout()

    @staticmethod
    @postprocess
    def dimensionality_reduction_2d(test_set: Dataset, result: Union[Result, None]):
        pass

    @staticmethod
    @postprocess
    def dimensionality_reduction_3d(test_set: Dataset, result: Union[Result, None]):
        pass

    @staticmethod
    @postprocess
    def tsne(test_set: Dataset, result: Union[Result, None], types: list, title=None):
        """
        针对不同的故障类型，使用T-SNE降维,使特征可视化
        :param title:
        :param types: 故障类型
        :param test_set: 故障诊断测试集
        :param result: 高维特征
        :return:
        """
        from fastphm.util.FeatureReduction import tsne
        # 生成二维特征
        if result is None:  # 对测试集做tsne
            features = test_set.x.reshape(test_set.x.shape[0], -1)
            features_t = tsne(features)
        else:  # 对结果做tsne
            features = result.y_hat
            features_t = tsne(features)
        y = test_set.y.reshape(-1)

        # 绘制散点图
        for i in range(len(types)):
            plt.scatter(features_t[y == i, 0], features_t[y == i, 1], s=10, label=types[i])

        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        if title is not None:
            plt.title(title)
