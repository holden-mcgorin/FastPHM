from __future__ import annotations
import numpy as np
from numpy import ndarray


class Dataset:
    def __init__(self, x: ndarray = None, y: ndarray = None, z: ndarray = None,
                 sub_label_map: dict = None, name: str = None):
        self.name = name
        self.__x = x  # 特征Feature
        self.__y = y  # 标签Label
        self.__z = z  # 已运行时间Time（s）
        self.sub_label_map = sub_label_map  # y的{标签：索引}字典,当y是多标签的时候才有用

        self.__validate(x, y, z)

    def __len__(self):
        return self.__x.shape[0]

    def __copy__(self):
        return Dataset(
            x=self.__x.copy() if self.__x is not None else None,
            y=self.__y.copy() if self.__y is not None else None,
            z=self.__z.copy() if self.__z is not None else None,
            sub_label_map=self.sub_label_map.copy() if self.sub_label_map is not None else None,
            name=self.name if self.name is not None else None,
        )

    def clear(self):
        self.__x = None
        self.__y = None
        self.__z = None
        self.sub_label_map = None

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def z(self):
        return self.__z

    @property
    def num_label(self):
        """
        :return:标签类别数量（兼容单标签与多标签）
        """
        return len(self.sub_label_map)

    def add(self, x: ndarray, y: ndarray, z: ndarray):
        """
        此数据集加数据
        :param x:
        :param y:
        :param z:
        :return:
        """
        self.__validate(x, y, z)

        if self.__x is None:
            self.__x = x
            self.__y = y
            self.__z = z
        else:
            self.__x = np.vstack((self.__x, x))
            self.__y = np.vstack((self.__y, y))
            self.__z = np.vstack((self.__z, z))

    def append(self, another_dataset: Dataset):
        """
        此数据集与另一个数据集合并
        :param another_dataset:
        :return:
        """
        # 添加数据
        if another_dataset.x is not None and another_dataset.y is not None:
            self.add(another_dataset.x, another_dataset.y, another_dataset.z)
            self.sub_label_map = another_dataset.sub_label_map

        # 添加数据集名称
        if another_dataset.name is not None:
            if self.name is not None:
                self.name = self.name + ';' + another_dataset.name
            else:
                self.name = another_dataset.name

    def split(self, ratio):
        """
        此数据集分裂为两个子数据集
        :param ratio:
        :return:
        """
        num_samples = self.__x.shape[0]
        indices = np.random.permutation(num_samples)

        train_size = int(ratio * num_samples)
        upper_indices = indices[:train_size]
        lower_indices = indices[train_size:]
        upper = Dataset(self.__x[upper_indices], self.__y[upper_indices], self.__z[upper_indices],
                        self.sub_label_map, self.name)
        lower = Dataset(self.__x[lower_indices], self.__y[lower_indices], self.__z[lower_indices],
                        self.sub_label_map, self.name)
        return upper, lower

    def add_sub_label(self, name: str, label_data: ndarray):
        """
        添加子标签
        :param name:
        :param label_data:
        :return:
        """
        if label_data.shape[0] != self.y.shape[0]:
            raise Exception(f'标签行数不匹配，源标签行数：{self.y.shape[0]}，目标标签行数：{label_data.shape[0]}')
        self.sub_label_map[name] = [self.y.shape[1], self.y.shape[1] + label_data.shape[1]]
        self.__y = np.hstack((self.__y, label_data))

    def get_sub_label(self, name: str = None):
        """
        获取子标签
        :param name:
        :return:
        """
        if name is not None:
            index = self.sub_label_map[name]
            return self.y[:, index[0]:index[1]]
        else:
            result = []
            for indices in self.sub_label_map.values():
                result.append(self.y[:, indices[0]:indices[1]])
            return tuple(result)

    def split_sub_label(self):
        """
        按子标签分裂数据集，即保持x、z不变，按y分裂
        :return:
        """
        results = []
        for key, indices in self.sub_label_map.items():
            sub_label_map = {key: [0, indices[1] - indices[0]]}
            results.append(
                Dataset(self.__x, self.__y[:, indices[0]: indices[1]], self.__z, sub_label_map, self.name))
        return tuple(results)

    def __validate(self, x, y, z):
        """
        验证数据集是否合法
        :param x:
        :param y:
        :param z:
        :return:
        """
        if x is None and y is None and z is None:
            return
        if x is not None and y is not None and z is not None:
            assert (x.shape[0] == y.shape[0] and y.shape[0] == z.shape[
                0]), f"x的行数：{x.shape[0]}, 与y的行数：{y.shape[0]},与z的行数：{z.shape[0]} 不相等"
            # if x.shape[0] != y.shape[0] or y.shape[0] != z.shape[0]:
            #     raise Exception('x的行数：', x.shape[0], '与y的行数：', y.shape[0], '与z的行数：', z.shape[0], ' 不相等')
        else:
            raise Exception('x、y、z需要同时初始化或同时为None')

        if self.sub_label_map is None:
            self.sub_label_map = {}
