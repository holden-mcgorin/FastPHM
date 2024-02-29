import os
import re
import pandas as pd
from rulframework.data_manager.raw_data.ABCDataLoader import ABCDataLoader
from rulframework.entity.Bearing import Bearing


class XJTUDataLoader(ABCDataLoader):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
            condition_dir = os.path.join(root_dir, condition)
            for bearing_name in os.listdir(condition_dir):
                self.item_dict[bearing_name] = os.path.join(root_dir, condition, bearing_name)
        print('成功登记以下轴承数据：')
        for key, value in self.item_dict.items():
            print(f"轴承: {key}，位置: {value}")

    def get_bearings_name(self) -> list:
        return list(self.item_dict.keys())

    def get_bearing(self, bearing_name: str, column=None) -> Bearing:
        """
        获取带有原始数据、轴承名的轴承对象
        :param column: 只取指定列数据（水平或垂直信号）
        :param bearing_name:轴承名
        :return:带有原始数据、轴承名的轴承对象
        """
        bearing = Bearing(bearing_name)
        bearing.raw_data = self.load_data(bearing_name, column)
        return bearing

    def load_data(self, item_name, column=None):
        """
        加载轴承的原始振动信号，返回包含raw_data的Bearing对象
        :param column: 只取指定列数据（水平或垂直信号）
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

        if column is not None:
            columns_names = bearing_raw_data.columns.tolist()
            for name in columns_names:
                if name != column:
                    bearing_raw_data.drop(name, axis=1, inplace=True)

        return bearing_raw_data

    # 自定义排序函数，从文件名中提取数字
    @staticmethod
    def __extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0
