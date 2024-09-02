import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from rulframework.data.loader.turbofan.ABCTurbofanLoader import ABCTurbofanLoader
from rulframework.entity.Turbofan import Turbofan


class PHM2008Loader(ABCTurbofanLoader):
    arr_min = {}
    arr_max = {}

    trajectories = {
        'train': 218,
        'test': 218,
        'final_test': 435
    }

    header = ['cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15',
              'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRF', 'NRc', 'BPR', 'farB',
              'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']

    def _register(self, root: str) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        file_dict = {}
        entity_dict = {}
        for filename in os.listdir(root):
            name = filename[:-4]
            if name in trajectories:
                file_dict[name] = os.path.join(root, filename)
                # 注册未分裂的完整数据
                entity_dict[name] = None
                for i in range(1, trajectories[name] + 1):
                    entity_name = name + '_' + str(i)
                    entity_dict[entity_name] = None
        return file_dict, entity_dict

    def _load(self, entity_name: str) -> DataFrame:
        # 如果字典为None（第一次查找）则加载
        if self._entity_dict[entity_name] is None:
            split = entity_name.split('_')
            prefix_raw = split[0]  # e.g. test
            if prefix_raw == 'final':
                prefix_raw = prefix_raw + '_test'

            df = pd.read_csv(os.path.join(self._root, prefix_raw + '.txt'), header=None, sep='\\s+')

            # 保存数据文件每列的最大值和最小值，用于归一化操作
            arr_min = np.min(df.values[:, 1:], axis=0)  # 每列的最小值
            arr_max = np.max(df.values[:, 1:], axis=0)  # 每列的最大值
            self.arr_min[prefix_raw] = arr_min
            self.arr_max[prefix_raw] = arr_max

            # 保存未分裂前的完整数据
            self._entity_dict[prefix_raw] = df.drop(df.columns[0], axis=1)

            # 按第一列分裂
            grouped = df.groupby(df.columns[0])
            dfs = {group: pd.DataFrame(data) for group, data in grouped}

            for i in range(1, trajectories[prefix_raw] + 1):
                self._entity_dict[prefix_raw + '_' + str(i)] = dfs[i].drop(dfs[i].columns[0], axis=1)

        return self._entity_dict[entity_name]

    def _assemble(self, entity_name: str, raw_data: DataFrame, columns_to_drop: [int] = None) -> Turbofan:
        turbofan = Turbofan(name=entity_name)

        # 增加表头
        raw_data.columns = header
        raw_data = self._load(entity_name)
        # 如果有column参数则删除对应列数据
        if columns_to_drop is not None:
            raw_data = raw_data.drop(raw_data.columns[columns_to_drop], axis=1)
        turbofan.raw_data = raw_data

        return turbofan
