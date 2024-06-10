import numpy as np
from numpy import ndarray


class Dataset:
    def __init__(self, x: ndarray = None, y: ndarray = None, name: str = None):
        self.name = name
        self.__x = x
        self.__y = y
        if x is not None and y is not None:
            if x.shape[0] != y.shape[0]:
                raise Exception('x的行数：', x.shape[0], '与y的行数：', y.shape[0], ' 不相等')

    def append(self, x: ndarray, y: ndarray):
        if x.shape[0] != y.shape[0]:
            raise Exception('x的行数：', x.shape[0], '与y的行数：', y.shape[0], ' 不相等')
        if self.__x is None:
            self.__x = x
            self.__y = y
        else:
            self.__x = np.vstack((self.__x, x))
            self.__y = np.vstack((self.__y, y))

    def split(self, ratio):
        num_samples = self.__x.shape[0]
        indices = np.random.permutation(num_samples)

        train_size = int(ratio * num_samples)
        upper_indices = indices[:train_size]
        lower_indices = indices[train_size:]
        upper = Dataset(self.__x[upper_indices], self.__y[upper_indices], self.name)
        lower = Dataset(self.__x[lower_indices], self.__y[lower_indices], self.name)
        return upper, lower

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y
