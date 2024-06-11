from __future__ import annotations
import numpy as np
from numpy import ndarray


class Dataset:
    def __init__(self, x: ndarray = None, y: ndarray = None, z: ndarray = None, name: str = None):
        self.name = name
        self.__x = x  # 特征
        self.__y = y  # 标签
        self.__z = z  # 已运行时间（s）

        self.__validate(x, y, z)

    @classmethod
    def __validate(cls, x, y, z):
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
            if x.shape[0] != y.shape[0] or y.shape[0] != z.shape[0]:
                raise Exception('x的行数：', x.shape[0], '与y的行数：', y.shape[0], '与z的行数：', z.shape[0], ' 不相等')
        else:
            raise Exception('x、y、z需要同时初始化或同时为None')

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def z(self):
        return self.__z

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
        if another_dataset.x is not None and another_dataset.y is not None:
            self.add(another_dataset.x, another_dataset.y, another_dataset.z)

        if self.name is not None and another_dataset.name is not None:
            self.name = self.name + ';' + another_dataset.name

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
        upper = Dataset(self.__x[upper_indices], self.__y[upper_indices], self.__z[upper_indices], self.name)
        lower = Dataset(self.__x[lower_indices], self.__y[lower_indices], self.__z[lower_indices], self.name)
        return upper, lower
