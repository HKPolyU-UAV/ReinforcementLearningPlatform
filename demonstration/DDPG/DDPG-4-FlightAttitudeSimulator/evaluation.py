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

from FlightAttitudeSimulator import FlightAttitudeSimulator as env


class Actor(nn.Module):
	def __init__(self, alpha, state_dim, action_dim, a_min, a_max):
		super(Actor, self).__init__()
		# self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.device = 'cpu'
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.a_min = torch.tensor(a_min, dtype=torch.float).to(self.device)
		self.a_max = torch.tensor(a_max, dtype=torch.float).to(self.device)
		self.off = (self.a_min + self.a_max) / 2.0
		self.gain = self.a_max - self.off
		self.fc1 = nn.Linear(self.state_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.mu = nn.Linear(256, self.action_dim)
		self.initialization()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
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
		x = self.gain * x + self.off
		return x

	def evaluate(self, s):
		t_state = torch.tensor(s, dtype=torch.float)
		mu = self.forward(t_state).cpu().detach().numpy().flatten()
		return mu


if __name__ == '__main__':
	optPath = './datasave/net/'
	env = env(0.)
	eval_net = Actor(state_dim=env.state_dim, action_dim=env.action_dim, a_min=env.action_range[:, 0], a_max=env.action_range[:, 1], alpha=1e-4)
	eval_net.load_state_dict(torch.load(optPath + 'target_actor'))

	n = 10
	for _ in range(n):
		env.reset_random()
		sumr = 0

		while not env.is_terminal:
			c = cv.waitKey(1)
			env.current_state = env.next_state.copy()
			action_from_actor = eval_net.evaluate(env.current_state)
			env.step_update(action_from_actor)
			env.visualization()

			sumr += env.reward
		print('Cumulative reward:', round(sumr, 3))
		print()
