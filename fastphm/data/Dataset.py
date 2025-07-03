from __future__ import annotations

import copy
from typing import Tuple, List, Union

import numpy as np
from numpy import ndarray


class Dataset:
    def __init__(self, name: str = None,
                 x: ndarray = None, y: ndarray = None, z: ndarray = None,
                 label_map: dict = None, entity_map: dict = None):
        """
        每个entity获取连续的一片空间
        :param name:
        :param x:
        :param y:
        :param z:
        :param label_map:
        :param entity_map:
        """
        self.name = name

        self.x = x  # 输入（特征）
        self.y = y  # 输出（标签）
        self.z = z  # 已运行时间Time（s）

        # 针对y的 e.g. {'label_1': (start, end)}
        self.label_map = label_map if label_map is not None else {}
        # 针对x的 e.g. {'entity_1': (start, end)}
        self.entity_map = entity_map if entity_map is not None else {}

        self.__validate(x, y, z)

    @property
    def num_label(self):
        """
        :return:标签类别数量（兼容单标签与多标签）
        """
        return len(self.label_map)

    @property
    def num_entity(self):
        """
        :return:标签类别数量（兼容单标签与多标签）
        """
        return len(self.entity_map)

    def append_entity(self, another_dataset: Dataset) -> None:
        """
        原地操作，合并另一个 Dataset。
        若 entity 名称重复，则将数据拼接到该实体末尾，并更新所有后续 entity 的起止位置。
        """
        # 合并的数据集为空，直接跳过
        if another_dataset.x is None:
            return

        # 当前数据集为空，直接替换
        if self.x is None:
            self.name = another_dataset.name
            self.x = another_dataset.x
            self.y = another_dataset.y
            self.z = another_dataset.z
            self.label_map = another_dataset.label_map
            self.entity_map = another_dataset.entity_map
            return

        # 确保标签类别一致
        if set(self.label_map.keys()) != set(another_dataset.label_map.keys()):
            raise ValueError("label_map's keys are not the same, cannot merge.")

        for entity, (start_new, end_new) in another_dataset.entity_map.items():
            data_x = another_dataset.x[start_new:end_new]
            data_y = another_dataset.y[start_new:end_new]
            data_z = another_dataset.z[start_new:end_new]
            new_len = end_new - start_new

            if entity in self.entity_map:
                start_old, end_old = self.entity_map[entity]
                # 插入到旧 entity 的末尾
                insert_pos = end_old

                # 插入数据
                self.x = np.insert(self.x, insert_pos, data_x, axis=0)
                self.y = np.insert(self.y, insert_pos, data_y, axis=0)
                self.z = np.insert(self.z, insert_pos, data_z, axis=0)

                # 更新该 entity 的映射
                self.entity_map[entity] = (start_old, end_old + new_len)

                # 更新所有在该 entity 之后的 entity 映射
                for key in self.entity_map:
                    if key != entity:
                        s, e = self.entity_map[key]
                        if s >= end_old:
                            self.entity_map[key] = (s + new_len, e + new_len)
            else:
                # 不冲突则直接追加
                start_new_idx = self.x.shape[0]
                self.x = np.concatenate([self.x, data_x], axis=0)
                self.y = np.concatenate([self.y, data_y], axis=0)
                self.z = np.concatenate([self.z, data_z], axis=0)
                self.entity_map[entity] = (start_new_idx, start_new_idx + new_len)

        self.name = f"{self.name}; {another_dataset.name}"

    def append_label(self, another_dataset: Dataset) -> None:
        """
        添加额外的标签
        """
        # 合并的数据集为空，直接跳过
        if another_dataset.x is None:
            return

        # 当前数据集为空，直接替换
        if self.x is None:
            self.name = another_dataset.name
            self.x = another_dataset.x
            self.y = another_dataset.y
            self.z = another_dataset.z
            self.label_map = another_dataset.label_map
            self.entity_map = another_dataset.entity_map
            return

        if another_dataset.y.shape[0] != self.y.shape[0]:
            raise Exception(f'标签行数不匹配，源标签行数：{self.y.shape[0]}，目标标签行数：{another_dataset.y.shape[0]}')

        another_datasets = another_dataset.split_by_label()
        for another_dataset in another_datasets:
            self.label_map[next(iter(another_dataset.label_map))] = (
                self.y.shape[1], self.y.shape[1] + another_dataset.y.shape[1])
            self.y = np.hstack((self.y, another_dataset.y))

    def split_by_ratio(self, ratio) -> Tuple[Dataset, Dataset]:
        """
        将数据集按比例拆分为两个子集
        :param ratio: 0~1之间的比例，表示第一个子集所占比例
        :return: 两个新的 Dataset 实例
        """

        # 先分为多个只包含单个entity的Dataset，再逐个按比例分裂
        uppers = []
        lowers = []
        for entity_dataset in self.split_by_entity():
            upper, lower = entity_dataset.__split(ratio)
            uppers.append(upper)
            lowers.append(lower)

        # 合并
        upper_result = Dataset()
        for upper in uppers:
            upper_result.append_entity(upper)
        lower_result = Dataset()
        for lower in lowers:
            lower_result.append_entity(lower)

        return upper_result, lower_result

    def split_by_label(self) -> Tuple[Dataset, ...]:
        """
        按子标签分裂数据集，即保持 x、z 不变，按 y 分裂
        :return: 每个子标签一个 Dataset
        """

        results = []
        for key, (start, end) in self.label_map.items():
            sub_label_map = {key: (0, end - start)}
            results.append(
                Dataset(
                    x=self.x,
                    y=self.y[:, start:end],
                    z=self.z,
                    label_map=sub_label_map,
                    entity_map=self.entity_map,  # 保留原始实体信息
                    name=f"{self.name}"
                )
            )
        return tuple(results)

    def split_by_entity(self) -> Tuple[Dataset, ...]:
        """
        按 entity_map 中的每个实体拆分数据集
        :return: 每个实体一个 Dataset
        """
        results = []
        for key, (start, end) in self.entity_map.items():
            x_part = self.x[start:end]
            y_part = self.y[start:end]
            z_part = self.z[start:end]

            # entity_map 只保留当前实体
            sub_entity_map = {key: (0, end - start)}

            results.append(
                Dataset(
                    x=x_part,
                    y=y_part,
                    z=z_part,
                    label_map=self.label_map,  # 所有实体共享同一个 label_map
                    entity_map=sub_entity_map,
                    name=f"{key}"
                )
            )
        return tuple(results)

    def select_by_features(self, indices: Union[List[int], int], squeeze=True) -> Dataset:
        """
        选择x中的指定列，生成新的Dataset
        :param squeeze:
        :param indices:
        :return:
        """
        if self.x.ndim != 3:
            raise Exception('仅支持维度为3的输入')
        # 当输入是单个索引时统一为索引列表
        if isinstance(indices, int):
            indices = [indices]

        new_x = self.x[:, :, indices]
        if squeeze:
            new_x = new_x.squeeze()

        return Dataset(
            x=new_x,
            y=self.y,
            z=self.z,
            label_map=copy.deepcopy(self.label_map),
            entity_map=copy.deepcopy(self.entity_map),
            name=self.name
        )

    def add(self, another_dataset: Dataset) -> None:
        """
        原地操作
        默认是 append_entity
        :param another_dataset:
        :return:
        """
        self.append_entity(another_dataset)

    def get(self, entity_name) -> Dataset:
        """
        非原地操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        return Dataset(
            x=self.x[start:end],
            y=self.y[start:end],
            z=self.z[start:end],
            label_map=self.label_map,
            entity_map={entity_name: (0, end - start)},
            name=entity_name
        )

    def remove(self, entity_name: str) -> None:
        """
        原地删除操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        shift = end - start
        # 生成新的entity_map
        if end == self.x.shape[0]:  # x中最后的entity
            del self.entity_map[entity_name]
        else:
            last = end
            while True:
                flag = False
                for k, (s, e) in self.entity_map.items():
                    if s == last:  # 找到下一个entity
                        self.entity_map[k] = (s - shift, e - shift)
                        last = e
                        flag = True
                        break
                if not flag:
                    break
            del self.entity_map[entity_name]

        self.x = np.concatenate([self.x[:start], self.x[end:]], axis=0),
        self.y = np.concatenate([self.y[:start], self.y[end:]], axis=0),
        self.z = np.concatenate([self.z[:start], self.z[end:]], axis=0),
        self.name.replace(entity_name, '')

    def include(self, entity_names: Union[str, List[str]]) -> Dataset:
        """
        非原地批量选择
        :param entity_names:
        :return:
        """
        entity_names = [entity_names] if isinstance(entity_names, str) else entity_names
        datasets = []
        for entity_name in entity_names:
            datasets.append(self.get(entity_name))

        new_dataset = datasets[0]
        for i in range(1, len(datasets)):
            new_dataset.append_entity(datasets[i])
        return new_dataset

    def exclude(self, entity_names: Union[str, List[str]]) -> Dataset:
        entity_names = [entity_names] if isinstance(entity_names, str) else entity_names
        new_dataset = self
        for entity_name in entity_names:
            new_dataset = new_dataset.__exclude(entity_name)
        return new_dataset

    def __exclude(self, entity_name: str) -> Dataset:
        """
        非原地删除操作
        :param entity_name:
        :return:
        """
        start, end = self.entity_map[entity_name]
        shift = end - start
        new_entity_map = copy.deepcopy(self.entity_map)
        # 生成新的entity_map
        if end == self.x.shape[0]:  # x中最后的entity
            del new_entity_map[entity_name]
        else:
            last = end
            while True:
                flag = False
                for k, (s, e) in new_entity_map.items():
                    if s == last:  # 找到下一个entity
                        new_entity_map[k] = (s - shift, e - shift)
                        last = e
                        flag = True
                        break
                if not flag:
                    break
            del new_entity_map[entity_name]
        new_name = copy.copy(self.name).replace(entity_name, '')

        return Dataset(
            x=np.concatenate([self.x[:start], self.x[end:]], axis=0),
            y=np.concatenate([self.y[:start], self.y[end:]], axis=0),
            z=np.concatenate([self.z[:start], self.z[end:]], axis=0),
            label_map=copy.deepcopy(self.label_map),
            entity_map=new_entity_map,
            name=new_name
        )

    def clear(self) -> None:
        """
        原地操作
        :return:
        """
        self.name = None
        self.x = None
        self.y = None
        self.z = None
        self.label_map = {}
        self.entity_map = {}

    def __add(self, x: ndarray, y: ndarray, z: ndarray) -> None:
        """
        此数据集加数据
        :param x:
        :param y:
        :param z:
        :return:
        """
        self.__validate(x, y, z)

        if self.x is None:
            self.x = x
            self.y = y
            self.z = z
        else:
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))
            self.z = np.vstack((self.z, z))

    def __split(self, ratio) -> Tuple[Dataset, Dataset]:
        """
        当数据集中只有一个entity时由概率分裂
        :param ratio:
        :return:
        """
        num_samples = self.x.shape[0]
        indices = np.random.permutation(num_samples)

        train_size = int(ratio * num_samples)
        upper_indices = indices[:train_size]
        lower_indices = indices[train_size:]

        upper_entity_map = {self.name: (0, train_size)}
        lower_entity_map = {self.name: (0, num_samples - train_size)}

        upper = Dataset(x=self.x[upper_indices], y=self.y[upper_indices], z=self.z[upper_indices],
                        label_map=self.label_map, entity_map=upper_entity_map, name=self.name)
        lower = Dataset(x=self.x[lower_indices], y=self.y[lower_indices], z=self.z[lower_indices],
                        label_map=self.label_map, entity_map=lower_entity_map, name=self.name)

        return upper, lower

    def __validate(self, x, y, z) -> None:
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

        if self.label_map is None:
            self.label_map = {}
        if self.entity_map is None:
            self.entity_map = {}

        # 分裂时产生空的数据集
        if self.x.shape[0] == 0:
            self.clear()

    def __len__(self):
        return self.x.shape[0]

    def __copy__(self):
        return Dataset(name=self.name if self.name is not None else None,
                       x=self.x.copy() if self.x is not None else None,
                       y=self.y.copy() if self.y is not None else None,
                       z=self.z.copy() if self.z is not None else None,
                       label_map=self.label_map.copy() if self.label_map is not None else None,
                       entity_map=self.entity_map.copy() if self.label_map is not None else None
                       )

# # 测试用例
# import numpy as np
#
# # 假设你的 Dataset 类已定义，包含 split_by_label 和 split_by_entity 方法
#
# # 构造一个虚拟数据集
# x = np.random.rand(10, 5)  # 10 条样本，每条5维特征
# y = np.random.rand(10, 6)  # 每条样本有6个标签值
# z = np.arange(10)  # 运行时间可以简单设为 0~9
#
# label_map = {
#     'A': (0, 2),  # y 的第0列到第2列（不含2）
#     'B': (2, 4),  # 第2列到第4列
#     'C': (4, 6),  # 第4列到第6列
# }
#
# entity_map = {
#     'E1': (0, 4),  # 第0~3行
#     'E2': (4, 7),  # 第4~6行
#     'E3': (7, 10),  # 第7~9行
# }
#
# dataset = Dataset(
#     name="TestData",
#     x=x,
#     y=y,
#     z=z,
#     label_map=label_map,
#     entity_map=entity_map
# )
#
# # ==== 测试 split_by_label ====
# print("=== Split by Label ===")
# label_splits = dataset.split_by_label()
# for i, ds in enumerate(label_splits):
#     print(f"Sub-dataset {i}:")
#     print("  Name:", ds.name)
#     print("  y shape:", ds._Dataset__y.shape)  # 访问私有变量
#     print("  label_map:", ds.label_map)
#     print()
#
# # ==== 测试 split_by_entity ====
# print("=== Split by Entity ===")
# entity_splits = dataset.split_by_entity()
# for i, ds in enumerate(entity_splits):
#     print(f"Sub-dataset {i}:")
#     print("  Name:", ds.name)
#     print("  x shape:", ds._Dataset__x.shape)
#     print("  entity_map:", ds.entity_map)
#     print()
#
# a, b = dataset.split_by_ratio(0.3)
# c = dataset.get('E1')
# d = dataset.remove('E2')


# # 构造原始数据集 A（有 entity1 和 entity2）
# x1 = np.array([[1], [2], [3], [4]])
# y1 = np.array([[10], [20], [30], [40]])
# z1 = np.array([[0], [0], [1], [1]])
# label_map = {'RUL': [0, 1]}
# entity_map1 = {'entity1': (0, 2), 'entity2': (2, 4)}
# A = Dataset(x=x1, y=y1, z=z1, label_map=label_map, entity_map=entity_map1, name="A")
#
# # 构造另一个数据集 B（含 entity1，和新的 entity3）
# x2 = np.array([[5], [6], [7], [8]])
# y2 = np.array([[50], [60], [70], [80]])
# z2 = np.array([[0], [0], [1], [1]])
# entity_map2 = {'entity1': (0, 2), 'entity3': (2, 4)}
# B = Dataset(x=x2, y=y2, z=z2, label_map=label_map, entity_map=entity_map2, name="B")
#
# # 合并 B 到 A
# A.append_entity(B)
#
# # 检查合并后数据
# print("x:")
# print(A.x)
# print("entity_map:")
# print(A.entity_map)
#
# # 期望结果：
# # entity1: 原 [0,2] + 新 [4,6] → [0,4]
# # entity2: 被插入后推迟 → [4,6]
# # entity3: 接在末尾 → [6,8]
# expected_x = np.array([[1], [2], [5], [6], [3], [4], [7], [8]])
# expected_entity_map = {
#     'entity1': (0, 4),
#     'entity2': (4, 6),
#     'entity3': (6, 8)
# }
#
# assert np.array_equal(A.x, expected_x)
# assert A.entity_map == expected_entity_map
#
# print("✅ 测试通过 append_entity (冲突合并)")
