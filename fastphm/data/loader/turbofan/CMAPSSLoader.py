import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from fastphm.data.loader.turbofan.ABCTurbofanLoader import ABCTurbofanLoader
from fastphm.entity.Turbofan import Fault, Turbofan, Condition
from fastphm.system.Logger import Logger


class CMAPSSLoader(ABCTurbofanLoader):
    """
    归一化参数设置
    """
    # 按照什么分组归一化（归一化的最大值和最小值来源）
    # 可选项："self"、"train"、"condition" 通常来说，self预测误差较大，其余差异不大
    normalization_from = 'condition'

    # 归一化模式
    # 可选项：'[0,1]'、'[-1,1]'、'None' 通常来说，该设置对预测的影响不大
    normalization_mode = '[0,1]'

    # setting1离散化参数（根据该值划分6种工况）
    # （low, high）-> val
    ranges = [
        (0.0, 0.003, 0.0),
        (9.998, 10.008, 10.0),
        (19.998, 20.008, 20.0),
        (24.998, 25.008, 25.0),
        (34.998, 35.008, 35.0),
        (41.998, 42.008, 42.0)
    ]

    # 存储每个数据集每个工况的最大最小值，e.g. group_min['FD002']['0.0']
    group_min = {'FD001': {},
                 'FD002': {},
                 'FD003': {},
                 'FD004': {}}
    group_max = {'FD001': {},
                 'FD002': {},
                 'FD003': {},
                 'FD004': {}}

    """
    硬数据
    """
    header = ['cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30',
              'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRF',
              'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

    num_trajectories = {
        'FD001_train': 100,
        'FD002_train': 260,
        'FD003_train': 100,
        'FD004_train': 249,
        'FD001_test': 100,
        'FD002_test': 259,
        'FD003_test': 100,
        'FD004_test': 248,
        'FD001_RUL': 100,
        'FD002_RUL': 259,
        'FD003_RUL': 100,
        'FD004_RUL': 248
    }

    faults = {
        'FD001': [Fault.HPC],
        'FD002': [Fault.HPC],
        'FD003': [Fault.HPC, Fault.Fan],
        'FD004': [Fault.HPC, Fault.Fan]
    }

    conditions = {
        'FD001': Condition.ONE,
        'FD002': Condition.SIX,
        'FD003': Condition.ONE,
        'FD004': Condition.SIX
    }

    min_cycle = {
        'FD001_train': 128,
        'FD002_train': 128,
        'FD003_train': 145,
        'FD004_train': 128,
        'FD001_test': 31,
        'FD002_test': 21,
        'FD003_test': 38,
        'FD004_test': 19,
    }

    max_cycle = {
        'FD001_train': 362,
        'FD002_train': 378,
        'FD003_train': 525,
        'FD004_train': 543,
        'FD001_test': 303,
        'FD002_test': 267,
        'FD003_test': 475,
        'FD004_test': 486,
    }

    def batch_load(self, trajectory: str, columns_to_drop: [int] = None) -> [Turbofan]:
        """
        批量加载数据：例如批量加载 FD001_train
        :param trajectory: 子数据集名称
        :param columns_to_drop: 需要剔除的列的序号
        :return:
        """
        entities = []
        num_trajectory = CMAPSSLoader.num_trajectories[trajectory]
        for i in range(1, num_trajectory + 1):
            entities.append(self(trajectory + '_' + str(i), columns_to_drop))

        return entities

    def _register(self, root) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        """
        记录数据文件位置
        :param root:
        :return:
        """
        file_dict = {}
        entity_dict = {}
        for filename in os.listdir(root):
            raw_name = filename[:-4]
            split = raw_name.split('_')
            if len(split) < 2:  # 非数据文件则跳过
                continue
            post_name = split[1] + '_' + split[0]
            if post_name in self.num_trajectories:
                file_dict[post_name] = os.path.join(root, filename)

                # 注册数据
                entity_dict[post_name] = None  # 注册完整数据
                entity_dict[post_name + '_norm'] = None  # 注册完整归一化数据
                for i in range(1, self.num_trajectories[post_name] + 1):  # 注册单个发动机数据
                    entity_dict[post_name + '_' + str(i)] = None
                    entity_dict[post_name + '_' + str(i) + '_norm'] = None
        return file_dict, entity_dict

    def _load(self, entity_name) -> DataFrame:
        """
        加载完整数据文件进入内存
        :param entity_name:
        :return:
        """
        # 如果字典已被加载过则直接返回
        if self._entity_dict[entity_name] is not None:
            return self._entity_dict[entity_name]

        # 否则第一次查找则加载
        split = entity_name.split('_')
        raw_name = split[1] + '_' + split[0]  # e.g. test_FD001
        post_name = split[0] + '_' + split[1]  # e.g. FD001_test

        # 读取数据文件并设置表头
        df = pd.read_csv(os.path.join(self._root, raw_name + '.txt'), header=None, sep='\\s+')
        df.columns = ['num'] + self.header

        # 保存完整数据
        self._entity_dict[post_name] = df
        # 保存单个实体的数据
        grouped = df.groupby(df.columns[0])
        dfs = {group: pd.DataFrame(data) for group, data in grouped}
        for i in range(1, self.num_trajectories[post_name] + 1):
            self._entity_dict[post_name + '_' + str(i)] = dfs[i].drop(dfs[i].columns[0], axis=1)

        # 若读取的发动机属于测试集需要额外读取RUL文件
        if split[1] == 'test':
            df = pd.read_csv(os.path.join(self._root, 'RUL_' + split[0] + '.txt'), header=None, sep='\\s+')
            for i in range(1, self.num_trajectories[split[0] + '_RUL'] + 1):
                self._entity_dict[split[0] + '_RUL_' + str(i)] = int(df.iloc[i - 1, 0])

        return self._entity_dict[entity_name]

    def _assemble(self, entity_name: str, raw_data: DataFrame, columns_to_drop: [int] = None) -> Turbofan:
        """
        生成涡扇发动机对象
        :param entity_name:
        :param raw_data:
        :param columns_to_drop:
        :return:
        """
        split = entity_name.split('_')

        turbofan = Turbofan(name=entity_name)
        turbofan.fault_type = self.faults[split[0]]
        turbofan.condition = self.conditions[split[0]]

        # 设置发动机的RUL（测试集发动机RUL从文件中读取；训练集发动机RUL为0）
        if len(split) == 3:  # 仅当获取的是单个发动机而不是整个子数据集时
            if split[1] == 'train':
                turbofan.rul = 0
                turbofan.life = raw_data.shape[0]
            if split[1] == 'test':
                turbofan.rul = self._entity_dict[split[0] + '_RUL_' + split[2]]
                turbofan.life = raw_data.shape[0] + turbofan.rul

        feature_data = self._norm(entity_name)
        # 如果有column参数则删除对应列数据
        if columns_to_drop is not None:
            raw_data = raw_data.drop(raw_data.columns[columns_to_drop], axis=1)
            feature_data = feature_data.drop(feature_data.columns[columns_to_drop], axis=1)
        turbofan.raw_data = raw_data
        turbofan.feature_data = feature_data

        return turbofan

    def _norm(self, entity_name: str) -> DataFrame:
        """
        生成归一化的子数据集整体数据
        :param entity_name:
        :return:
        """
        # 如果字典已被加载过则直接返回
        if self._entity_dict[entity_name + '_norm'] is not None:
            return self._entity_dict[entity_name + '_norm']

        split = entity_name.split('_')
        raw_df = self._load(split[0] + '_' + split[1])
        df = raw_df.copy()

        if self.normalization_from == "condition":

            # setting1离散化
            def discretize(val):
                for low, high, target in self.ranges:
                    if low <= val <= high:
                        return target
                Logger.warning(f'During the discretization, '
                               f'the value {val} exceeded all intervals and has been returned as None')
                return None

            # 添加两个辅助列
            if split[0] in ['FD001', 'FD003']:
                df['condition'] = 1
            elif split[0] in ['FD002', 'FD004']:
                df['condition'] = df['setting1'].apply(discretize)
            else:
                raise KeyError(f"{split[0]} does not exist")
            df['original_index'] = df.index
            grouped = []
            for name, group in df.groupby('condition'):
                features = group.iloc[:, 2:-2]  # 去掉前两列和最后两个辅助列
                if split[1] == 'test':
                    self._norm(split[0] + '_train')
                if split[1] == 'train':
                    self.group_min[split[0]][name] = features.min()
                    self.group_max[split[0]][name] = features.max()
                normalized = self._cal(features, self.group_min[split[0]][name], self.group_max[split[0]][name])
                group.update(normalized)  # 更新归一化后的特征列
                grouped.append(group)
            result = pd.concat(grouped).sort_values('original_index').drop(columns='original_index').drop(
                columns='condition')
        elif self.normalization_from == "self":
            result = self._cal(df, df.min(), df.max())
        elif self.normalization_from == "train":
            train_df = self._load(split[0] + '_train')
            result = self._cal(df, train_df.min(), train_df.max())
        else:
            raise KeyError(f"parameter 'normalization_from' now is {self.normalization_from}, which is invalid!")

        # 保存完整数据
        self._entity_dict[split[0] + '_' + split[1] + '_norm'] = result
        # 保存单个实体的数据
        grouped = result.groupby(result.columns[0])
        dfs = {group: pd.DataFrame(data) for group, data in grouped}
        for i in range(1, self.num_trajectories[split[0] + '_' + split[1]] + 1):
            self._entity_dict[split[0] + '_' + split[1] + '_' + str(i) + '_norm'] = dfs[i].drop(dfs[i].columns[0],
                                                                                                axis=1)

        return self._entity_dict[entity_name + '_norm']

    def _cal(self, features, min_val, max_val):
        """
        归一化计算算法
        :param features:
        :param min_val:
        :param max_val:
        :return:
        """
        if self.normalization_mode == '[0,1]':
            normalized = (features - min_val) / (max_val - min_val)
        elif self.normalization_mode == '[-1,1]':
            normalized = 2 * (features - min_val) / (max_val - min_val) - 1
        elif self.normalization_mode == 'None':
            normalized = features
        else:
            raise KeyError(f"parameter 'normalization_mode' now is {self.normalization_mode}, which is invalid!")

        return normalized
