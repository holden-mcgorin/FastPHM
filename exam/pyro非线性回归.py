import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression

# 生成合成数据
X, y = make_regression(n_features=1, bias=150., noise=5., random_state=108)

# 转换为 PyTorch 张量
X_ = torch.tensor(X, dtype=torch.float)
y_ = torch.tensor((y ** 3) / 100000. + 10., dtype=torch.float)
# 对张量 y_ 中的每个元素执行四舍五入，并将结果截断为非负值
y_.round_().clamp_(min=0)

# 可视化数据
plt.scatter(X_, y_)
plt.ylabel('y_test')
plt.xlabel('x_test')
plt.show()


# 定义模型
def model(features, counts):
    N, P = features.shape
    scale = pyro.sample("scale", dist.LogNormal(0, 1))
    coef = pyro.sample("coef", dist.Normal(0, scale).expand([P]).to_event(1))
    rate = pyro.deterministic("rate", torch.nn.functional.softplus(coef @ features.T))
    concentration = pyro.sample("concentration", dist.LogNormal(0, 1))
    with pyro.plate("bins", N):
        return pyro.sample("counts", dist.GammaPoisson(concentration, rate), obs=counts)


# 运行 HMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=500)
mcmc.run(X_, y_)

# 提取后验样本
samples = mcmc.get_samples()

# 打印后验样本的形状
for k, v in samples.items():
    print(f"{k}: {tuple(v.shape)}")

# 进行预测
predictive = Predictive(model, samples)(X_, None)

# 打印预测结果的形状
for k, v in predictive.items():
    print(f"{k}: {tuple(v.shape)}")


# 准备预测结果的 DataFrame
def prepare_counts_df(predictive):
    counts = predictive['counts'].numpy()
    counts_mean = counts.mean(axis=0)
    counts_std = counts.std(axis=0)
    counts_df = pd.DataFrame({
        "feat": X_.squeeze(),
        "mean": counts_mean,
        "high": counts_mean + counts_std,
        "low": counts_mean - counts_std,
    })

    return counts_df.sort_values(by=['feat'])


# 可视化预测结果
counts_df = prepare_counts_df(predictive)

plt.scatter(X_, y_, c='r')
plt.ylabel('y_test')
plt.xlabel('x_test')
plt.plot(counts_df['feat'], counts_df['mean'])
plt.fill_between(counts_df['feat'], counts_df['high'], counts_df['low'], alpha=0.5)
plt.show()
