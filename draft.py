import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.distributions import  MultivariateNormal, Normal


# a_min = np.array([-1])
# a_max = np.array([1])
#
# noise_policy = a_max - a_min
#
# action = torch.zeros((10))
# noise = torch.randn_like(action)
# noise_policy = noise * noise_policy
#
# print(noise)
# print(noise_policy)
# sigma = np.array([0.1, 1, 3])
# a = np.random.multivariate_normal(np.zeros_like(sigma), np.diag(sigma**2))
# print(a)

# mean = torch.tensor([0., 0.])
# cov = torch.tensor([[1., 0.], [0., 1.]])
# mvn = MultivariateNormal(mean, covariance_matrix=cov)
# x = mvn.sample((2,1))
# print(x)

mean = torch.zeros(3)
std = torch.tensor([1, 10., 100])
dis = Normal(mean, std)
a = dis.sample((5, 1)).squeeze()

clip = torch.tensor([1, 2, 3])
noise = torch.maximum(torch.minimum(a, clip), -clip)

print(a)
print(noise)
