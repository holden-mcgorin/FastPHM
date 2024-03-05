import pyro.distributions as dist
import torch
from pandas import DataFrame
from pyro.infer import MCMC, NUTS
import pyro
from torch import nn
import random

from rulframework.model.ABCModel import ABCModel


class PyroModel(ABCModel):

    def __init__(self, model: nn.Module) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_model = model.to(self.device).double()  # 确定性模型
        self.samples = None  # 模型参数的样本（由训练时MCMC生成）

    def predict(self, input_data: list) -> list:
        # todo 下面这段代码有硬编码
        idx = random.randint(0, len(self.samples["fc1.weight"]) - 1)
        weight1_sample = self.samples["fc1.weight"][idx].unsqueeze(0)
        bias1_sample = self.samples["fc1.bias"][idx].unsqueeze(0)
        weight2_sample = self.samples["fc2.weight"][idx].unsqueeze(0)
        bias2_sample = self.samples["fc2.bias"][idx].unsqueeze(0)

        # 抽取样本代入模型参数
        self.torch_model.fc1.weight = torch.nn.Parameter(weight1_sample.squeeze())
        self.torch_model.fc1.bias = torch.nn.Parameter(bias1_sample.squeeze())
        self.torch_model.fc2.weight = torch.nn.Parameter(weight2_sample.squeeze())
        self.torch_model.fc2.bias = torch.nn.Parameter(bias2_sample.squeeze())

        # 进行一次神经网络的前向传播
        x = torch.tensor(input_data, dtype=torch.float64).to(self.device)
        return self.torch_model(x).tolist()

    def predict_uncertainty(self, input_data: list) -> list:
        return super().predict_uncertainty(input_data)

    # 构建贝叶斯神经网络模型
    def pyro_model(self, x, y):
        # 定义网络结构的参数为随机变量（参数的先验分布为均值0、标准差1的正态分布）
        for name, param in self.torch_model.named_parameters():
            loc = torch.zeros_like(param.data)  # 均值为 0
            scale = torch.ones_like(param.data)  # 标准差为 1
            pyro.sample(name, dist.Normal(loc, scale))
        # 前向传播
        y_pred = self.torch_model(x)
        # 观测数据
        pyro.sample('obs', dist.Normal(y_pred.squeeze(), 0.1), obs=y)

    def train(self, train_data_x: DataFrame, train_data_y: DataFrame, num_epochs: int = 1000):
        """
        贝叶斯推断
        :param train_data_x:
        :param train_data_y:
        :param num_epochs:
        :return:
        """
        # 使用 NUTS 进行 MCMC 推断
        nuts_kernel = NUTS(self.pyro_model)
        mcmc = MCMC(nuts_kernel, num_samples=num_epochs, warmup_steps=200)
        x = torch.tensor(train_data_x.values, dtype=torch.float64).to(self.device)
        y = torch.tensor(train_data_y.values, dtype=torch.float64).to(self.device)
        mcmc.run(x.unsqueeze(1), y)
        self.samples = mcmc.get_samples()
