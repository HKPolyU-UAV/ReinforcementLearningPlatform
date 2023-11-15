import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


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
sigma = np.array([0.1, 1, 3])
a = np.random.multivariate_normal(np.zeros_like(sigma), np.diag(sigma**2))
print(a)