import os
import re

import pandas as pd

from core.Bearing.Bearing import Bearing
from core.DataLoader.DataLoader import DataLoader


class XJTUDataLoader(DataLoader):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
            condition_dir = os.path.join(root_dir, condition)
            for bearing_name in os.listdir(condition_dir):
                self.item_dict[bearing_name] = os.path.join(root_dir, condition, bearing_name)
        for key, value in self.item_dict.items():
            print(f"轴承: {key}，位置: {value}")

    # 自定义排序函数，从文件名中提取数字
    def __extract_number(self, file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0

    def load_data(self, item_name):
        """
        加载轴承的原始振动信号，返回包含raw_data的Bearing对象
        :param item_name:
        :return: Bearing对象（包含raw_data)
        """
        bearing_raw_data = pd.DataFrame()
        bearing_dir = self.item_dict[item_name]
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
        return bearing_raw_data


if __name__ == '__main__':
    bearing = Bearing('Bearing1_1')
    data_loader = XJTUDataLoader('D://data//dataset//XJTU-SY_Bearing_Datasets')
    bearing.raw_data = data_loader.load_data(bearing.name)
    print(bearing.raw_data)
    bearing.raw_data_figure()
