import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import numpy as np
import matplotlib.pyplot as plt


# 生成示例时序数据
def generate_data(num_points=50):
    x = torch.linspace(0, 10, num_points)
    y = torch.sin(x) + torch.randn(num_points) * 0.1  # 加入噪声
    return x, y


x_train, y_train = generate_data()


# 定义一个简单的多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(10, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 构建贝叶斯神经网络模型
def model(x, y):
    mlp = MLP()
    # 定义网络结构的参数为随机变量
    for name, param in mlp.named_parameters():
        pyro.sample(name, dist.Normal(0, 1))
    # 前向传播
    y_pred = mlp(x)
    # 观测数据
    pyro.sample('obs', dist.Normal(y_pred.squeeze(), 0.1), obs=y)


# 使用 NUTS 进行 MCMC 推断
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(x_train.unsqueeze(1), y_train)

# 从后验分布中抽取样本
posterior_samples = mcmc.get_samples()


# 使用后验样本进行预测
def predict(x, posterior_samples):
    mlp = MLP()
    y_preds = torch.stack([mlp(x).squeeze() for _ in range(len(posterior_samples))])
    print(y_preds)
    return y_preds.mean(0), y_preds.std(0)


x_test = torch.linspace(0, 10, 100)
y_pred_mean, y_pred_std = predict(x_test.unsqueeze(1), posterior_samples)

# 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), label='Training Data')
plt.plot(x_test.tolist(), y_pred_mean.tolist(), color='red', label='Mean Prediction')
plt.fill_between(x_test.tolist(), y_pred_mean.tolist(),
                 y_pred_mean.tolist(), color='red', alpha=0.2, label='Uncertainty')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Bayesian Neural Network Prediction with Uncertainty')
plt.legend()
plt.show()
