import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建一个实例化的神经网络
model = SimpleNN()

# 定义观测数据
x_data = torch.randn(100, 10)
y_data = torch.randn(100, 1)

# 定义模型
def model(x_data, y_data):
    # 权重参数的先验分布
    fc1_weight_prior = dist.Normal(torch.zeros_like(model.fc1.weight), 1.0).to_event(2)
    fc2_weight_prior = dist.Normal(torch.zeros_like(model.fc2.weight), 1.0).to_event(2)

    # 将权重参数设置为先验分布
    priors = {'fc1.weight': fc1_weight_prior, 'fc2.weight': fc2_weight_prior}

    # 通过模型定义条件分布
    lifted_module = pyro.random_module("module", model, priors)
    lifted_model = lifted_module()
    with pyro.plate("data", len(x_data)):
        prediction_mean = lifted_model(x_data).squeeze(-1)
        pyro.sample("obs", dist.Normal(prediction_mean, 0.1), obs=y_data.squeeze(-1))

# 使用 NUTS 算法进行 HMC 采样
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(x_data, y_data)

# 获取采样结果
posterior_samples = mcmc.get_samples()

# 打印参数后验分布的统计信息
for param_name in posterior_samples.keys():
    print(param_name)
    print("Mean:", posterior_samples[param_name].mean(dim=0))
    print("Std:", posterior_samples[param_name].std(dim=0))
    print()
