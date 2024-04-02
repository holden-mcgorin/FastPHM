import pyro.distributions as dist
import torch
from pandas import DataFrame
from pyro.infer import MCMC, NUTS, HMC
import pyro
from torch import nn
import random

from rulframework.model.ABCModel import ABCModel


class PyroModel(ABCModel):

    def __init__(self, model: nn.Module) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_model = model.to(device=self.device, dtype=torch.float64)  # 确定性模型
        self.samples = None  # 模型参数的样本（由训练时MCMC生成）

    def pyro_model(self, x, y):
        # 定义网络结构的参数为随机变量（参数的先验分布为均值0、标准差1的正态分布）
        for name, param in self.torch_model.named_parameters():
            loc = torch.zeros_like(param.data)  # 均值为 0
            scale = torch.ones_like(param.data)  # 标准差为 1
            pyro.sample(name, dist.Normal(loc, scale))
        # 前向传播
        y_pred = self.torch_model(x)
        # 观测数据
        pyro.sample('obs', dist.Normal(y_pred, 0.1), obs=y)

    def train(self, train_data_x: DataFrame, train_data_y: DataFrame, num_epochs: int = 1000):
        """
        贝叶斯推断
        :param train_data_x:
        :param train_data_y:
        :param num_epochs:在这里为从马尔科夫链中采样的个数
        :return:
        """
        # 使用 HMC 进行 MCMC 推断
        hmc_kernel = HMC(self.pyro_model)
        mcmc = MCMC(hmc_kernel, num_samples=num_epochs, warmup_steps=200)
        x = torch.tensor(train_data_x.values, dtype=torch.float64, device=self.device)
        y = torch.tensor(train_data_y.values, dtype=torch.float64, device=self.device)
        mcmc.run(x, y)
        self.samples = mcmc.get_samples()

    # 构建贝叶斯神经网络模型
    def predict(self, input_data: list) -> list:
        """
        预测一次
        :param input_data: 输入数据
        :return: 预测结果（一次），可能是单步预测也可能是多步预测
        """
        # todo 下面这段代码有硬编码
        idx_1 = random.randint(0, len(self.samples["fc1.weight"]) - 1)
        idx_2 = random.randint(0, len(self.samples["fc1.bias"]) - 1)
        idx_3 = random.randint(0, len(self.samples["fc2.weight"]) - 1)
        idx_4 = random.randint(0, len(self.samples["fc2.bias"]) - 1)
        weight1_sample = self.samples["fc1.weight"][idx_1]
        bias1_sample = self.samples["fc1.bias"][idx_2]
        weight2_sample = self.samples["fc2.weight"][idx_3]
        bias2_sample = self.samples["fc2.bias"][idx_4]

        # 抽取样本代入模型参数
        self.torch_model.fc1.weight = torch.nn.Parameter(weight1_sample)
        self.torch_model.fc1.bias = torch.nn.Parameter(bias1_sample)
        self.torch_model.fc2.weight = torch.nn.Parameter(weight2_sample)
        self.torch_model.fc2.bias = torch.nn.Parameter(bias2_sample)

        # 进行一次神经网络的前向传播
        x = torch.tensor(input_data, dtype=torch.float64, device=self.device)
        return self.torch_model(x).tolist()

    def plot_loss(self):
        pass
