import torch
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import HMC, MCMC


# 定义一个简单的概率模型
def model():
    # 定义参数的先验分布
    mu = pyro.sample("mu", dist.Normal(0, 1))
    sigma = pyro.sample("sigma", dist.Uniform(0, 1))

    # 定义观测数据的条件分布
    with pyro.plate("data", size=100):
        obs = pyro.sample("obs", dist.Normal(mu, sigma))


# 使用 HMC 进行推断
nuts_kernel = HMC(model, step_size=0.1, num_steps=4)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run()

# 获取参数的后验分布
posterior_samples = mcmc.get_samples()

# 打印参数后验分布的统计信息
for param_name in posterior_samples.keys():
    print(param_name)
    print("Mean:", posterior_samples[param_name].mean(dim=0))
    print("Std:", posterior_samples[param_name].std(dim=0))
    print()
