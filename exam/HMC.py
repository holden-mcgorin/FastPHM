import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import numpy as np
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):
    # x_test = torch.normal(0, 1, (num_examples, len(w)))
    x = torch.linspace(0, 10, num_examples)
    y = x * w + b
    y += torch.normal(0, 1, y.shape)
    return x, y.reshape((-1, 1))


# 生成示例时序数据
def generate_data(num_points=50):
    x = torch.linspace(0, 10, num_points)
    y = torch.sin(x) + torch.randn(num_points) * 0.1  # 加入噪声
    return x, y


x_train, y_train = synthetic_data(torch.tensor([2.]), 10, 50)


# 构建贝叶斯神经网络模型
def model(x, y):
    # 定义网络结构
    w = pyro.sample('w', dist.Normal(0, 1))  # 权重
    b = pyro.sample('b', dist.Normal(0, 1))  # 偏置

    # 前向传播
    y_pred = w * x + b

    # 观测数据
    pyro.sample('obs', dist.Normal(y_pred, 0.1), obs=y)


# 使用 NUTS 进行 MCMC 推断
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=200)
mcmc.run(x_train, y_train)

# 从后验分布中抽取样本
posterior_samples = mcmc.get_samples()


# 使用后验样本进行预测
def predict(x, posterior_samples):
    w_samples = posterior_samples['w']
    b_samples = posterior_samples['b']
    # y_preds = torch.stack([w * __x + b for w, b in zip(w_samples, b_samples)])
    print(w_samples.mean(0), b_samples.std(0))
    return w_samples.mean(0), b_samples.std(0)


x_test = torch.linspace(0, 10, 100)
y_pred_mean, y_pred_std = predict(x_test, posterior_samples)

# 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), label='Training Data')
plt.plot(x_test.numpy(), y_pred_mean.numpy(), color='red', label='Mean Prediction')
plt.fill_between(x_test.numpy(), y_pred_mean.numpy() - 2 * y_pred_std.numpy(),
                 y_pred_mean.numpy() + 2 * y_pred_std.numpy(), color='red', alpha=0.2, label='Uncertainty')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Bayesian Neural Network Prediction with Uncertainty')
plt.legend()
plt.show()
