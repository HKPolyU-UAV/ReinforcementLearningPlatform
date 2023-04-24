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
	action_space = [[-3.0, -2.5, -2.0], [-1.5, -1.0]]
	batch_action_number = np.array([[-3.0, -1.0], [-2.0, -1.0], [-2.5, -1.5]], dtype=float)
	row = batch_action_number.shape[0]
	col = batch_action_number.shape[1]
	res = [[-1] * col for _ in range(row)]
	for i in range(batch_action_number.shape[0]):
		for j, a in enumerate(batch_action_number[i]):
			res[i][j] = action_space[j].index(a)
	print(torch.tensor(res).float())
