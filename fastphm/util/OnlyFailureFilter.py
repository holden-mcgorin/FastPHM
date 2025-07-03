import numpy as np

from fastphm.data.Dataset import Dataset
from fastphm.model.Result import Result

"""
作用：
仅取失效后数据，即过滤掉失效前的数据
"""


class OnlyFailureFilter:
    def __call__(self, test_set: Dataset, result: Result, enable=True):
        if not enable:
            return test_set, result
        index = np.where(test_set.y != 1)[0]
        sub_test_set = Dataset(x=test_set.x[index, :], y=test_set.y[index, :], z=test_set.z[index, :],
                               label_map=test_set.label_map, name=test_set.name, entity_map=test_set.entity_map)

        sub_result = result.__copy__()
        sub_result.outputs = sub_result.outputs[index, :]
        return sub_test_set, sub_result
