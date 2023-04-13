# import os
# import math
# import cv2 as cv
import atexit
import time

import numpy as np
import torch

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


if __name__ == '__main__':
	action_space = [2, 2, 5, 10]
	dist = Categorical(probs=nn.utils.rnn.pad_sequence([torch.ones(action_space[i], dtype=torch.float32) / action_space[i] for i in range(4)]).T)
	a = dist.sample()
	print(a)
	print(torch.mean(dist.log_prob(a)))
	pass
