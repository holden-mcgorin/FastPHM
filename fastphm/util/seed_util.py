# 文件名：seed_util.py

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    设置所有可控随机种子，确保实验可复现

    :param seed: 随机种子值
    :param deterministic: 是否使用确定性操作（会降低性能）
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[Seed]  Random seed set to {seed} (deterministic={deterministic})")


def seed_worker(worker_id):
    """
    DataLoader 中 worker 进程的随机种子初始化函数

    :param worker_id: worker进程编号
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# if __name__ == '__main__':
#     # from seed_util import set_seed
#
#     set_seed(2025)
#     from torch.utils.data import DataLoader, Dataset
#     # from seed_util import seed_worker
#     import torch
#
#     g = torch.Generator()
#     g.manual_seed(2025)
#     dataset = Dataset()
#
#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=True,
#         worker_init_fn=seed_worker,  # 设置worker种子
#         generator=g  # 设置全局随机生成器
#     )
