# import atexit
# import os
# import time
#
# import mpi4py.MPI
# import numpy as np
# import math
# from environment.config.xml_write import xml_cfg
# import random
# import cv2 as cv
import atexit
import time

import numpy as np
import torch

from common.common_func import *
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# from environment.envs.pathplanning.bezier import Bezier
# import gym
import torch.nn as nn
# import collections
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Normal
# from torch.distributions import Categorical
# from torch.distributions import MultivariateNormal
# # from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
# from multiprocessing import shared_memory
from torch.distributions import Normal
from torch.distributions import MultivariateNormal


if __name__ == '__main__':
	a = np.atleast_1d([1, 2, 3]).astype(np.float32)
	print(a)
	a = np.hstack((a, 1))
	print(a)
	pass
