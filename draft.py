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
	def __init__(self, nAction = 2, nOut=None):
		super(SoftmaxActor, self).__init__()
		self.nAction = nAction
		# TODO
		if nOut is None:
			self.nOut = [3, 3]
		self.index = [0]
		for i in range(nAction):
			self.index.append(self.index[i] + self.nOut[i])
		self.fc1 = nn.Linear(2, 64)
		self.fc2 = nn.Linear(64, 64)
		# self.out1 = nn.Linear(64, nOut)
		# self.out2 = nn.Linear(64, nOut)
		# TODO
		# self.out = [nn.Linear(64, 3) for i in range(nAction)]
		self.out = nn.Linear(64, sum(self.nOut))
		self.fc3 = nn.Linear(64, 3)

	def forward(self, xx):
		# xx = torch.FloatTensor(xx)
		xx = torch.tanh(self.fc1(xx))
		xx = torch.tanh(self.fc2(xx))
		# TODO
		xx = self.out(xx)

		a_prob = []
		for i in range(self.nAction):
			# a_prob.append(func.softmax(self.out[i](xx), dim=1).T)
			# TODO
			# a_prob.append(self.out[i](xx).T)
			a_prob.append(func.softmax(xx[:, self.index[i]:self.index[i + 1]], dim=1).T)
		return nn.utils.rnn.pad_sequence(a_prob).T

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
	actor1 = SoftmaxActor()
	actor2 = SoftmaxActor()
	actor3 = SoftmaxActor()
	actor4 = SoftmaxActor()
	actor5 = SoftmaxActor()

	actor2.load_state_dict(actor1.state_dict())
	actor3.load_state_dict(actor1.state_dict())
	actor4.load_state_dict(actor1.state_dict())
	actor5.load_state_dict(actor1.state_dict())

	xx = torch.randn((1, 2))
	y1 = actor1(xx).cpu().detach().numpy()
	y2 = actor2(xx).cpu().detach().numpy()
	y3 = actor3(xx).cpu().detach().numpy()
	y4 = actor4(xx).cpu().detach().numpy()
	y5 = actor5(xx).cpu().detach().numpy()
	print(y1)
	print(y2)
	print(y3)
	print(y4)
	print(y5)
	pass
