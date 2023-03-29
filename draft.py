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
from multiprocessing import shared_memory


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim):
		super(PPOActorCritic, self).__init__()
		self.action_dim = _action_dim
		self.state_dim = _state_dim
		self.actor = nn.Sequential(
			nn.Linear(4, self.action_dim),		#
			nn.Tanh()
		)
		self.critic = nn.Sequential(
			nn.Linear(4, self.action_dim),  #
			nn.ReLU6(),
			nn.Linear(4, 1),  #
		)

	def forward(self):
		raise NotImplementedError


# def param_send(net: PPOActorCritic):
# 	for p, shape in zip(net.parameters(), net_shape):
# 		_param = p.detach().cpu().numpy()
# 		_share_m = shared_memory.SharedMemory(create=True, size=_param.nbytes)
# 		share_memory.append(_share_m)
#
# 		_share_p = np.ndarray(shape=shape, dtype=np.float32, buffer=_share_m.buf)
# 		_share_p[:] = _param[:]
# 		share_param.append(_share_p)

def get_net_param_shape(net: PPOActorCritic):
	n = []
	for p in net.parameters():
		n.append(p.detach().cpu().numpy().shape)
	return n

def param_get(share_memory, shape):
	net = PPOActorCritic(3, 2).to('cpu')
	print('before')
	print_net_param(net)

	for _s, _param, _shape in zip(share_memory, net.parameters(), shape):
		temp_np = np.ndarray(shape=_shape, dtype=np.float32, buffer=_s.buf)
		# print('in....')
		# print(temp_np)	# OK 没问题的
		with torch.no_grad():
			_param.copy_(torch.tensor(temp_np))

	print('copy finished...')
	print('after')
	print_net_param(net)


def print_net_param(net:PPOActorCritic):
	print('----')
	for p in net.parameters():
		_param = p.detach().cpu().numpy()
		print(_param)
	print('----')


if __name__ == '__main__':
	# mp.set_start_method('spawn', True)
	# mp.set_start_method('fork', True)
	share_memory = []
	share_param = []
	net = PPOActorCritic(3, 2).to('cuda:0')

	print('original net:....')
	print_net_param(net)


	net_shape = get_net_param_shape(net)
	print(net_shape)

	for shape, param in zip(net_shape, net.parameters()):
		numpy_buf = np.zeros(shape=shape, dtype=np.float32)
		sm = shared_memory.SharedMemory(create=True, size=numpy_buf.nbytes)		# 网络参数对应的共享内存
		share_buf = np.ndarray(numpy_buf.shape, numpy_buf.dtype, sm.buf)		# 指向该共享内存的 numpy
		share_buf[:] = param.detach().cpu().numpy()[:]
		share_memory.append(sm)
		share_param.append(share_buf)

	p1 = mp.Process(target=param_get, args=(share_memory, net_shape))
	p1.start()

	net = PPOActorCritic(3, 2).to('cuda:0')


	p1.join()
	# print_net_param(net2)

	# print(share_param)
	# p1.terminate()
	# p2.join()
	@atexit.register
	def clean():
		for _s in share_memory:
			_s.close()
			_s.unlink()
