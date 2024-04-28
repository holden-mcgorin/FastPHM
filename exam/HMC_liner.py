import torch
import pyro
import pyro.distributions as dist
from matplotlib import pyplot as plt
from pyro.infer import MCMC, NUTS


# 定义 synthetic_data 函数
def synthetic_data(w, b, train_num):
    x = torch.normal(10, 3, [train_num])
    y = x * w + b
    y += torch.normal(0, 1, y.shape)
    return x, y


# 生成合成数据
x_test, y_test = synthetic_data(2., 100, 100)


# 定义模型
def model(x, y):
    w = pyro.sample("w", dist.Normal(0, 1))
    b = pyro.sample("b", dist.Normal(0, 1))
    mu = x * w + b
    sigma = pyro.sample("sigma", dist.Uniform(0, 10))
    with pyro.plate("data", len(x)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=y)


# 运行 HMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=500)
mcmc.run(x_test, y_test)

# 提取后验样本
posterior_samples = mcmc.get_samples()

# 打印后验分布的统计信息
print(posterior_samples["w"].mean())
print(posterior_samples["b"].mean())
print(posterior_samples["sigma"].mean())

# 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(x_test.numpy(), y_test.numpy(), label='Training Data')
# plt.plot(x_test.numpy(), y_pred_mean.numpy(), color='red', label='Mean Prediction')
# plt.fill_between(x_test.numpy(), y_pred_mean.numpy() - 2 * y_pred_std.numpy(),
#                  y_pred_mean.numpy() + 2 * y_pred_std.numpy(), color='red', alpha=0.2, label='Uncertainty')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Bayesian Neural Network Prediction with Uncertainty')
plt.legend()
plt.show()
