import os
import re
from typing import Dict

import pandas as pd
from rulframework.data.raw.ABCDataLoader import ABCDataLoader
from rulframework.entity.Bearing import Bearing, BearingFault


class XJTUDataLoader(ABCDataLoader):
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
            'Bearing1_1': [BearingFault.OF],
            'Bearing1_2': [BearingFault.OF],
            'Bearing1_3': [BearingFault.OF],
            'Bearing1_4': [BearingFault.CF],
            'Bearing1_5': [BearingFault.IF, BearingFault.OF],
            'Bearing2_1': [BearingFault.IF],
            'Bearing2_2': [BearingFault.OF],
            'Bearing2_3': [BearingFault.CF],
            'Bearing2_4': [BearingFault.OF],
            'Bearing2_5': [BearingFault.OF],
            'Bearing3_1': [BearingFault.OF],
            'Bearing3_2': [BearingFault.IF, BearingFault.OF, BearingFault.CF, BearingFault.BF],
            'Bearing3_3': [BearingFault.IF],
            'Bearing3_4': [BearingFault.IF],
            'Bearing3_5': [BearingFault.OF],
        }
        return fault_type_dict

    def _build_item_dict(self, root) -> Dict[str, str]:
        item_dict = {}
        for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
            condition_dir = os.path.join(root, condition)
            for bearing_name in os.listdir(condition_dir):
                item_dict[bearing_name] = os.path.join(root, condition, bearing_name)
        return item_dict

    def _load_raw_data(self, item_name):
        """
        加载轴承的原始振动信号，返回包含raw_data的Bearing对象
        :param item_name:
        :return: Bearing对象（包含raw_data)
        """
        bearing_dir = self._item_dict[item_name]

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
