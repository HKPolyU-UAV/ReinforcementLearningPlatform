import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


a_min = [-1, 3, 5]
a_max = [1, 4, 7]

a = np.random.uniform(a_min, a_max)
print(a)
