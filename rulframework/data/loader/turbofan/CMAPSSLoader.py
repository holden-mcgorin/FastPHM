import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from rulframework.data.loader.turbofan.ABCTurbofanLoader import ABCTurbofanLoader
from rulframework.entity.Turbofan import Fault, Turbofan, Condition


class CMAPSSLoader(ABCTurbofanLoader):
    arr_min = {}
    arr_max = {}

    trajectories = {
        'FD001_train': 100,
        'FD002_train': 260,
        'FD003_train': 100,
        'FD004_train': 248,
        'FD001_test': 100,
        'FD002_test': 259,
        'FD003_test': 100,
        'FD004_test': 249
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

    header = ['cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15',
              'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRF', 'NRc', 'BPR', 'farB',
              'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

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

            df = pd.read_csv(os.path.join(self._root, raw_name + '.txt'), header=None, sep='\\s+')

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

            for i in range(1, self.trajectories[post_name] + 1):
                self._entity_dict[post_name + '_' + str(i)] = dfs[i].drop(dfs[i].columns[0], axis=1)

        return self._entity_dict[entity_name]

    def _assemble(self, entity_name: str, raw_data: DataFrame, columns_to_drop: [int] = None) -> Turbofan:
        split = entity_name.split('_')

        turbofan = Turbofan(name=entity_name)
        turbofan.fault_type = self.faults[split[0]]
        turbofan.condition = self.conditions[split[0]]

        # 增加表头
        raw_data.columns = self.header
        raw_data = self._load(entity_name)
        # 如果有column参数则删除对应列数据
        if columns_to_drop is not None:
            raw_data = raw_data.drop(raw_data.columns[columns_to_drop], axis=1)
        turbofan.raw_data = raw_data

        return turbofan
