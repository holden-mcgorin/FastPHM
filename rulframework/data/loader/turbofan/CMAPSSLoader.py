import os
from enum import Enum
from typing import Dict, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from rulframework.data.loader.turbofan.ABCTurbofanLoader import ABCTurbofanLoader
from rulframework.entity.Turbofan import Fault, Turbofan, Condition
from rulframework.system.Logger import Logger


class CMAPSSLoader(ABCTurbofanLoader):
    # 记录每个子数据集中的每个传感器的最小值和最大值，用于归一化处理
    arr_min = {}
    arr_max = {}
    normalization_mode = '[0,1]'  # 归一化模式，可选项：'[-1,1]'、'None'

    trajectories = {
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

    header = ['cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30',
              'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRF',
              'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

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

    def batch_load(self, trajectory: str, columns_to_drop: [int] = None) -> [Turbofan]:
        """
        批量加载数据：例如批量加载 FD001_train
        :param trajectory: 子数据集名称
        :param columns_to_drop: 需要剔除的列的序号
        :return:
        """
        entities = []
        num_trajectory = CMAPSSLoader.trajectories[trajectory]
        for i in range(1, num_trajectory + 1):
            entities.append(self(trajectory + '_' + str(i), columns_to_drop))

        return entities

    def _register(self, root) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        file_dict = {}
        entity_dict = {}
        for filename in os.listdir(root):
            raw_name = filename[:-4]
            split = raw_name.split('_')
            if len(split) < 2:
                continue
            post_name = split[1] + '_' + split[0]
            if post_name in self.trajectories:

                file_dict[post_name] = os.path.join(root, filename)
                # 注册未分裂的完整数据
                entity_dict[post_name] = None
                for i in range(1, self.trajectories[post_name] + 1):
                    entity_name = post_name + '_' + str(i)
                    entity_dict[entity_name] = None
        return file_dict, entity_dict

    def _load(self, entity_name) -> DataFrame:
        # 如果字典为None（第一次查找）则加载
        if self._entity_dict[entity_name] is None:
            split = entity_name.split('_')
            raw_name = split[1] + '_' + split[0]  # e.g. test_FD001
            post_name = split[0] + '_' + split[1]  # e.g. FD001_test

            # 读取数据文件并设置表头
            df = pd.read_csv(os.path.join(self._root, raw_name + '.txt'), header=None, sep='\\s+')
            df.columns = ['num'] + self.header

            # 保存数据文件每列的最大值和最小值，用于归一化操作
            arr_min = np.min(df.values[:, 1:], axis=0)  # 每列的最小值
            arr_max = np.max(df.values[:, 1:], axis=0)  # 每列的最大值
            self.arr_min[post_name] = arr_min
            self.arr_max[post_name] = arr_max

            # 按第一列分裂
            grouped = df.groupby(df.columns[0])
            dfs = {group: pd.DataFrame(data) for group, data in grouped}

            # 保存未分裂前的完整数据
            self._entity_dict[post_name] = df.drop(df.columns[0], axis=1)
            # 保存分裂后单个实体的数据
            for i in range(1, self.trajectories[post_name] + 1):
                self._entity_dict[post_name + '_' + str(i)] = dfs[i].drop(dfs[i].columns[0], axis=1)

            # 若读取的发动机属于测试集需要额外读取RUL文件
            if split[1] == 'test':
                df = pd.read_csv(os.path.join(self._root, 'RUL_' + split[0] + '.txt'), header=None, sep='\\s+')
                for i in range(1, self.trajectories[split[0] + '_RUL'] + 1):
                    self._entity_dict[split[0] + '_RUL_' + str(i)] = int(df.iloc[i - 1, 0])

        return self._entity_dict[entity_name]

    def _assemble(self, entity_name: str, raw_data: DataFrame, columns_to_drop: [int] = None) -> Turbofan:
        split = entity_name.split('_')

        turbofan = Turbofan(name=entity_name)
        turbofan.fault_type = self.faults[split[0]]
        turbofan.condition = self.conditions[split[0]]
        # 设置发动机的RUL（测试集发动机RUL从文件中读取；训练集发动机RUL为0）
        if split[1] == 'train':
            turbofan.rul = 0
            turbofan.life = raw_data.shape[0]
        if split[1] == 'test':
            turbofan.rul = self._entity_dict[split[0] + '_RUL_' + split[2]]
            turbofan.life = raw_data.shape[0] + turbofan.rul

        # 如果有column参数则删除对应列数据
        subset_name = split[0] + '_' + split[1]
        if columns_to_drop is not None:
            raw_data = raw_data.drop(raw_data.columns[columns_to_drop], axis=1)
            arr_min = np.delete(self.arr_min[subset_name], columns_to_drop)
            arr_max = np.delete(self.arr_max[subset_name], columns_to_drop)
        else:
            arr_min = self.arr_min[subset_name]
            arr_max = self.arr_max[subset_name]
        turbofan.raw_data = raw_data

        # 计算归一化数据（特征数据）
        if self.normalization_mode == 'None':
            return turbofan
        data = raw_data.values
        # 0到1归一化
        if self.normalization_mode == '[0,1]':
            arr_normalized = (data - arr_min) / (arr_max - arr_min)
        # -1到1归一化
        else:
            arr_normalized = 2 * (data - arr_min) / (arr_max - arr_min) - 1
        column_names_list = raw_data.columns.tolist()
        turbofan.feature_data = pd.DataFrame(arr_normalized, columns=column_names_list)

        return turbofan
