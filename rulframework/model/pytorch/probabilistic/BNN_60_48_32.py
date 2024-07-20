import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linear_BBB(nn.Module):
    """
        一层参数是服从正态分布的全连接层
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=1.):
        """
            参数的先验分布为均值0、方差20的正态分布
            rho = ρ, mu = μ
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # initialize weight samples (these will be calculated whenever the layer makes a outputs)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = Normal(2, prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(device)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(device)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w).to(device)
        b_log_prior = self.prior.log_prob(self.b).to(device)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with
        # respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class BNN_60_48_32(nn.Module):
    """
        由两层BBB构成的贝叶斯神经网络（中间有一层sigmoid）
    """

    def __init__(self, noise_tol=.1, prior_var=1.):
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.hidden = Linear_BBB(60, 48, prior_var=prior_var)
        self.out = Linear_BBB(48, 32, prior_var=prior_var)
        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        # 对数先验概率（log prior）是对先验分布取对数得到的结果
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input_value, target, samples):
        # 负证据下界作为损失函数
        # we calculate the negative elbo, which will be our loss function
        # initialize tensors
        outputs = torch.zeros(samples, target.shape[0], target.shape[1]).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_posts = torch.zeros(samples).to(device)
        log_likes = torch.zeros(samples).to(device)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input_value).reshape(target.shape[0], target.shape[1])  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.outputs()
        log_post = log_posts.outputs()
        log_like = log_likes.outputs()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss
