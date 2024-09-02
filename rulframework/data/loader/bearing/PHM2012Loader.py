import os
import re
from typing import Dict, Union

import pandas as pd
from pandas import DataFrame

from rulframework.data.loader.bearing.ABCBearingLoader import ABCBearingLoader


class PHM2012Loader(ABCBearingLoader):

    @property
    def frequency(self) -> int:
        return 25600

    @property
    def continuum(self) -> int:
        return 2560

    @property
    def span(self) -> int:
        return 10

    def _register(self, root: str) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        file_dict = {}
        entity_dict = {}
        for folder in ['Learning_set', 'Full_Test_Set']:
            folder_dir = os.path.join(root, folder)
            for bearing_name in os.listdir(folder_dir):
                file_dict[bearing_name] = os.path.join(root, folder, bearing_name)
                entity_dict[bearing_name] = None
        return file_dict, entity_dict

    def _load(self, entity_name) -> DataFrame:
        bearing_dir = self._file_dict[entity_name]

        # 仅获取加速度文件并排序（PHM2012还有温度传感器数据）
        files = os.listdir(bearing_dir)
        acc_files = [file for file in files if file.startswith('acc')]
        acc_files = sorted(acc_files, key=self.__extract_number)

        # 读取所有加速度csv文件数据保存在raw_data中
        dataframes = []
        if entity_name == 'Bearing1_4':  # Bearing1_4数据文件使用;作为分隔符
            for acc_file in acc_files:
                df = pd.read_csv(os.path.join(bearing_dir, acc_file), header=None, sep=';').iloc[:, -2:]
                dataframes.append(df)
        else:
            for acc_file in acc_files:  # 其他轴承数据文件
                df = pd.read_csv(os.path.join(bearing_dir, acc_file), header=None).iloc[:, -2:]
                dataframes.append(df)
        raw_data = pd.concat(dataframes, axis=0, ignore_index=True)

        # 规范列名
        column_names = list(raw_data.columns)
        column_names[0] = 'Horizontal Vibration'
        column_names[1] = 'Vertical Vibration'
        raw_data.columns = column_names

        return raw_data

    # 自定义排序函数，从文件名中提取数字
    @staticmethod
    def __extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0
