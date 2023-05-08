# import os
# import math
# import cv2 as cv
import atexit
import time
from itertools import product

import numpy as np
import torch
from matplotlib import pyplot as plt

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

    t = np.linspace(0, 10, 100)
    plt.plot(t, - t - np.tanh(2.5 * t) + 1)
    plt.show()


