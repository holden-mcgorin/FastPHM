import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import summary
from pyro.distributions import constraints
import pyro
import torch

pyro.set_rng_seed(101)

X, y = make_regression(n_features=1, bias=150., noise=5., random_state=108)

X_ = torch.tensor(X, dtype=torch.float)
y_ = torch.tensor((y ** 3) / 100000. + 10., dtype=torch.float)
y_.round_().clamp_(min=0)

plt.scatter(X_, y_)
plt.ylabel('y')
plt.xlabel('x')
plt.show()


def model(features, counts):
    N, P = features.shape
    scale = pyro.sample("scale", dist.LogNormal(0, 1))
    coef = pyro.sample("coef", dist.Normal(0, scale).expand([P]).to_event(1))
    rate = pyro.deterministic("rate", torch.nn.functional.softplus(coef @ features.T))
    concentration = pyro.sample("concentration", dist.LogNormal(0, 1))
    with pyro.plate("bins", N):
        return pyro.sample("counts", dist.GammaPoisson(concentration, rate), obs=counts)


nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=500)

mcmc.run(X_, y_)

samples = mcmc.get_samples()
for k, v in samples.items():
    print(f"{k}: {tuple(v.shape)}")

predictive = Predictive(model, samples)(X_, None)
for k, v in predictive.items():
    print(f"{k}: {tuple(v.shape)}")


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


counts_df = prepare_counts_df(predictive)

plt.scatter(X_, y_, c='r')
plt.ylabel('y')
plt.xlabel('x')
plt.plot(counts_df['feat'], counts_df['mean'])
plt.fill_between(counts_df['feat'], counts_df['high'], counts_df['low'], alpha=0.5)
plt.show()
