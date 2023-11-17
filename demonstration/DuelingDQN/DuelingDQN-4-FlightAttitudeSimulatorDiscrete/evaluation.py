import sys
import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from FlightAttitudeSimulatorDiscrete import FlightAttitudeSimulatorDiscrete as env


class DuelingNeuralNetwork(nn.Module):
	def __init__(self, state_dim=1, action_dim=1):
		"""
		:brief:             神经网络初始化
		:param state_dim:      输入维度
		:param action_dim:     输出维度
		"""
		super(DuelingNeuralNetwork, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(state_dim, 64)  # input -> hidden1
		self.fc2 = nn.Linear(64, 64)  # hidden1 -> hidden2
		self.value = nn.Linear(64, 1)

		assert action_dim <= 100, '动作空间过大，建议采用其他RL算法'
		self.advantage = nn.Linear(64, action_dim)
		# self.init()
		self.init_default()

		self.device = 'cpu'
		self.to(self.device)

	def init(self):
		torch.nn.init.orthogonal_(self.fc1.weight, gain=1)
		torch.nn.init.uniform_(self.fc1.bias, 0, 1)
		torch.nn.init.orthogonal_(self.fc2.weight, gain=1)
		torch.nn.init.uniform_(self.fc2.bias, 0, 1)
		torch.nn.init.orthogonal_(self.out.weight, gain=1)
		torch.nn.init.uniform_(self.out.bias, 0, 1)
		torch.nn.init.orthogonal_(self.value.weight, gain=1)
		torch.nn.init.uniform_(self.value.bias, 0, 1)
		torch.nn.init.orthogonal_(self.advantage.weight, gain=1)
		torch.nn.init.uniform_(self.advantage.bias, 0, 1)

	def init_default(self):
		self.fc1.reset_parameters()
		self.fc2.reset_parameters()
		self.value.reset_parameters()
		self.advantage.reset_parameters()

	def forward(self, _x):
		"""
		:brief:         神经网络前向传播
		:param _x:      输入网络层的张量
		:return:        网络的输出
		"""
		x = _x
		x = self.fc1(x)
		x = func.relu(x)
		x = self.fc2(x)
		x = func.relu(x)

		x1 = self.value(x)
		x2 = self.advantage(x)

		state_action_value = x1 + (x2 - x2.mean())
		return state_action_value

	def evaluate(self, s):
		t_state = torch.tensor(s).float().to(self.device)
		v = self.forward(t_state).cpu().detach().numpy()
		return np.argmax(v)


if __name__ == '__main__':
	optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/net/'
	env = env()
	eval_net = DuelingNeuralNetwork(state_dim=env.state_dim, action_dim=env.action_num[0])
	# torch.load(optPath + 'eval')
	eval_net.load_state_dict(torch.load(optPath + 'eval'))

	n = 10
	for _ in range(n):
		env.reset(random=True)
		sumr = 0
		while not env.is_terminal:
			c = cv.waitKey(1)
			env.current_state = env.next_state.copy()
			action_from_actor = eval_net.evaluate(env.current_state)
			action = np.array([env.action_space[0][action_from_actor]])
			env.step_update(action)
			env.visualization()

			sumr += env.reward
		print('Cumulative reward:', round(sumr, 3))
		print()
