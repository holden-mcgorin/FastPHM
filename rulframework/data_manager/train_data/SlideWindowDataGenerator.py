import os
import shutil

import pandas as pd
from pandas import DataFrame

from rulframework.data_manager.train_data.ABCDataGenerator import ABCDataGenerator


class SlideWindowDataGenerator(ABCDataGenerator):
    """
    利用硬盘空间分段缓存生成滑动窗口训练数据
    """
    def __init__(self, window_size: int, window_step: int = 1, temp_dir: str = './temp'):
        """
        @param window_size: 滑动窗口大小
        @param window_step: 滑动窗口每次滑动的步长
        @param temp_dir: 临时缓存数据的存储文件夹位置（最后会自动删除）
        """
        self.window_size = window_size
        self.window_step = window_step
        self.temp_dir = temp_dir

    def generate_data(self, source_data: DataFrame) -> DataFrame:
        # 计算切片分片结构(索引列表，例：0,8,16)
        index_list = []
        for i in range(0, self.window_size, 8):
            index_list.append(i)
        index_list.append(self.window_size)

        # 开始切片，生成临时文件
        temp_file_list = []
        for i in range(len(index_list) - 1):
            partial_window_data = self.__create_partial_window_data(source_data, index_list[i], index_list[i + 1])
            partial_window_data = partial_window_data[::self.window_step]  # 取步长
            temp_filename = str(index_list[i]) + '_' + str(index_list[i + 1]) + '.csv'
            self.__save_temp(partial_window_data, temp_filename)
            temp_file_list.append(temp_filename)
            print(f'created temp data: {temp_filename} ......', end='\r', flush=True)

        # 合并切片
        train_data = self.__merge_temp(temp_file_list)
        self.__delete_temp()

        # 重命名列名
        new_columns = list(range(len(train_data.columns)))
        train_data.columns = new_columns

        return train_data

    @staticmethod
    def __create_partial_window_data(source_data: DataFrame, begin: int, end: int):
        column_data = [source_data.iloc[:, 0].shift(-i) for i in range(begin, end)]
        partial_window_data = pd.concat(column_data, axis=1)
        return partial_window_data

    # 保存中间结果
    def __save_temp(self, dataframe: DataFrame, filename: str):
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        dataframe.to_csv(os.path.join(self.temp_dir, filename), index=False, sep=',', encoding='utf-8')
        return

    # 将中间结果合并,temp_file_list为temp_file文件名列表
    def __merge_temp(self, temp_file_list):
        train_data = pd.DataFrame()
        for file_name in temp_file_list:
            data = pd.read_csv(os.path.join(self.temp_dir, file_name))
            train_data = pd.concat([train_data, data], axis=1, ignore_index=True)
        train_data = train_data.dropna()  # 删除含nan值的列
        return train_data

    # 删除中间结果
    def __delete_temp(self):
        shutil.rmtree(self.temp_dir)
