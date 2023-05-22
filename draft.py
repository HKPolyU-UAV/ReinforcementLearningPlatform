# import os
# import math
# import cv2 as cv
import atexit
import time
from itertools import product

import numpy as np
import torch
from matplotlib import pyplot as plt
import sympy
from common.common_func import *
# import pandas as pd
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# import torch.multiprocessing as mp
# from multiprocessing import shared_memory
from torch.distributions import Uniform
from environment.envs.SecondOrderIntegration.SecondOrderIntegration import SecondOrderIntegration

if __name__ == '__main__':
    l = 0.35
    basePos = [1., 1.]
    x = np.linspace(0, 1, 100)
    gamma = 0.1
    interval1 = [1 if i < gamma else 0 for i in x]
    interval2 = [1 if i >= gamma else 0 for i in x]
    y = ((1 - x) ** 2 - 0.8) / 2 * interval1 + (- gamma * (x - gamma / 2)) * interval2
    # y = - x - np.tanh(2.5 * x) + 1
    plt.plot(x, y)
    plt.show()
