import os
import re
from typing import Dict

import pandas as pd
from rulframework.data_manager.raw.ABCDataLoader import ABCDataLoader


class XJTUDataLoader(ABCDataLoader):

    @property
    def span(self) -> int:
        return 32768

    def _build_item_dict(self, root) -> Dict[str, str]:
        item_dict = {}
        for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
            condition_dir = os.path.join(root, condition)
            for bearing_name in os.listdir(condition_dir):
                item_dict[bearing_name] = os.path.join(root, condition, bearing_name)
        return item_dict

    def _load(self, item_name, columns=None):
        """
        加载轴承的原始振动信号，返回包含raw_data的Bearing对象
        :param columns: 只取指定列数据（水平或垂直信号）
        :param item_name:
        :return: Bearing对象（包含raw_data)
        """
        bearing_raw_data = pd.DataFrame()
        bearing_dir = self._item_dict[item_name]

        files = sorted(os.listdir(bearing_dir), key=self.__extract_number)
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(bearing_dir, file_name)
                data = pd.read_csv(file_path)
                bearing_raw_data = pd.concat([bearing_raw_data, data], axis=0, ignore_index=True)

        # 规范列名
        bearing_raw_data.rename(columns={'Horizontal_vibration_signals': 'Horizontal Vibration',
                                         'Vertical_vibration_signals': 'Vertical Vibration'},
                                inplace=True)

        # 如果有column参数则仅取该列数据
        if columns is not None:
            columns_names = bearing_raw_data.columns.tolist()
            for name in columns_names:
                if name != columns:
                    bearing_raw_data.drop(name, axis=1, inplace=True)

        return bearing_raw_data

    # 自定义排序函数，从文件名中提取数字
    @staticmethod
    def __extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0
