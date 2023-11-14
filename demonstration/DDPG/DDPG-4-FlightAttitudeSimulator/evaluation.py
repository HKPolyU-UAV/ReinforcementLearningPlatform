import sys
import datetime
import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from FlightAttitudeSimulator import Flight_Attitude_Simulator as env


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, alpha=1e-4):
		super(Actor, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.fc1 = nn.Linear(self.state_dim, 128)
		self.fc2 = nn.Linear(128, 64)
		self.mu = nn.Linear(64, self.action_dim)
		self.initialization()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
		# self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.device = 'cpu'
		self.to(self.device)

	def initialization(self):
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		nn.init.uniform_(self.fc1.bias.data, -f1, f1)

		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		nn.init.uniform_(self.fc2.bias.data, -f2, f2)

		f3 = 0.003
		nn.init.uniform_(self.mu.weight.data, -f3, f3)
		nn.init.uniform_(self.mu.bias.data, -f3, f3)

	def forward(self, s):
		s = func.relu(self.fc1(s))
		s = func.relu(self.fc2(s))
		x = torch.tanh(self.mu(s))
		return x

	def evaluate(self, s):
		s = torch.tensor(s, dtype=torch.float).to(self.device)
		return self.forward(s).cpu().detach().numpy()


def action_linear_trans(action):
	# the action output
	linear_action = []
	for i in range(env.action_dim):
		a = min(max(action[i], -1), 1)
		maxa = env.action_range[i][1]
		mina = env.action_range[i][0]
		k = (maxa - mina) / 2
		b = (maxa + mina) / 2
		linear_action.append(k * a + b)
	return np.array(linear_action)


if __name__ == '__main__':
	optPath = './datasave/net/'
	env = env(0.)
	eval_net = Actor(state_dim=env.state_dim, action_dim=env.action_dim)
	eval_net.load_state_dict(torch.load(optPath + 'target_actor'))

	n = 10

	for _ in range(n):
		# env.reset()
		env.reset_random()
		sumr = 0

		while not env.is_terminal:
			c = cv.waitKey(1)
			env.current_state = env.next_state.copy()
			action_from_actor = eval_net.evaluate(env.current_state)
			action = action_linear_trans(action_from_actor)
			env.step_update(action)
			env.visualization()

			sumr += env.reward
		print('Cumulative reward:', round(sumr, 3))
		print()
