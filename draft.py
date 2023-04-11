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


class SoftmaxActor(nn.Module):
	def __init__(self, nAction = 4, nOut=None):
		super(SoftmaxActor, self).__init__()
		self.nAction = nAction
		if nOut is None:
			nOut = [4, 4, 3, 5]
		self.fc1 = nn.Linear(4, 64)
		self.fc2 = nn.Linear(64, 64)
		self.out = [nn.Linear(64, nOut[i]) for i in range(nAction)]
		self.initialization()

	def initialization(self):
		self.orthogonal_init(self.fc1)
		self.orthogonal_init(self.fc2)
		for i in range(self.nAction):
			self.orthogonal_init(self.out[i], gain=0.01)

	@staticmethod
	def orthogonal_init(layer, gain=1.0):
		nn.init.orthogonal_(layer.weight, gain=gain)
		nn.init.constant_(layer.bias, 0)

	def forward(self, xx):
		# xx = torch.FloatTensor(xx)
		xx = torch.tanh(self.fc1(xx))
		xx = torch.tanh(self.fc2(xx))
		a_prob = []
		for i in range(self.nAction):
			a_prob.append(func.softmax(self.out[i](xx), dim=1).T)
		return nn.utils.rnn.pad_sequence(a_prob).T
		# return a_prob

	def evaluate(self, xx):				# evaluate 默认是在测试情况下的函数，默认没有batch
		xx = torch.unsqueeze(xx, 0)
		a_prob = self.forward(xx)
		_a = torch.argmax(a_prob, dim=2)
		return _a

	def choose_action(self, xx):		# choose action 默认是在训练情况下的函数，默认有batch
		# xx = torch.unsqueeze(xx, 0)
		with torch.no_grad():
			dist = Categorical(probs=self.forward(xx))
			_a = dist.sample()
			_a_logprob = dist.log_prob(_a)
			_a_entropy = dist.entropy()
		return _a, torch.mean(_a_logprob, dim=1), torch.mean(_a_entropy, dim=1)


if __name__ == '__main__':
	# actor = SoftmaxActor()
	# xx1 = torch.randn((100, 4))
	# # y = actor(xx1)
	# # a = actor.evaluate(torch.squeeze(xx1))
	# a1, a1_lg_prob, a1_entropy = actor.choose_action(xx1)
	# # a1_lg_prob = torch.mean(a1_lg_prob, dim=1)
	# print(a1_lg_prob.size(), a1_entropy.size())
	#
	# xx2 = torch.randn((100, 4))
	# a2, a2_lg_prob, a2_entropy = actor.choose_action(xx2)
	# #
	# ratios = torch.exp(a2_lg_prob.detach() - a2_lg_prob.detach())
	# print(ratios.size())
	a1 = np.array([1.0]).astype(np.float32)
	a2 = np.array([1.0]).astype(np.float32)
	b = np.hstack((a1, a2))
	print(b)
	pass
