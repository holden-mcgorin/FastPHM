import os
import re
from typing import Dict, Union

import pandas as pd
from pandas import DataFrame

from rulframework.data.loader.bearing.ABCBearingLoader import ABCBearingLoader
from rulframework.entity.Bearing import Fault


class XJTULoader(ABCBearingLoader):
    @property
    def frequency(self) -> int:
        return 25600

    @property
    def span(self) -> int:
        return 60

    @property
    def continuum(self) -> int:
        return 32768

    @property
    def fault_type_dict(self) -> dict:
        fault_type_dict = {
            'Bearing1_1': [Fault.OF],
            'Bearing1_2': [Fault.OF],
            'Bearing1_3': [Fault.OF],
            'Bearing1_4': [Fault.CF],
            'Bearing1_5': [Fault.IF, Fault.OF],
            'Bearing2_1': [Fault.IF],
            'Bearing2_2': [Fault.OF],
            'Bearing2_3': [Fault.CF],
            'Bearing2_4': [Fault.OF],
            'Bearing2_5': [Fault.OF],
            'Bearing3_1': [Fault.OF],
            'Bearing3_2': [Fault.IF, Fault.OF, Fault.CF, Fault.BF],
            'Bearing3_3': [Fault.IF],
            'Bearing3_4': [Fault.IF],
            'Bearing3_5': [Fault.OF],
        }
        return fault_type_dict

    def _register(self, root: str) -> (Dict[str, str], Dict[str, Union[DataFrame, None]]):
        file_dict = {}
        entity_dict = {}
        for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
            condition_dir = os.path.join(root, condition)
            for bearing_name in os.listdir(condition_dir):
                file_dict[bearing_name] = os.path.join(root, condition, bearing_name)
                entity_dict[bearing_name] = None
        return file_dict, entity_dict

    def _load(self, entity_name) -> DataFrame:
        """
        加载轴承的原始振动信号，返回包含raw_data的Bearing对象
        :param entity_name:
        :return: Bearing对象（包含raw_data)
        """
        bearing_dir = self._file_dict[entity_name]

        # 读取csv数据并合并
        dataframes = []
        files = sorted(os.listdir(bearing_dir), key=self.__extract_number)
        for file_name in files:
            df = pd.read_csv(os.path.join(bearing_dir, file_name))
            dataframes.append(df)
        raw_data = pd.concat(dataframes, axis=0, ignore_index=True)

        # 规范列名
        raw_data.rename(columns={'Horizontal_vibration_signals': 'Horizontal Vibration',
                                 'Vertical_vibration_signals': 'Vertical Vibration'},
                        inplace=True)

        return raw_data

    # 自定义排序函数，从文件名中提取数字
    @staticmethod
    def __extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0
