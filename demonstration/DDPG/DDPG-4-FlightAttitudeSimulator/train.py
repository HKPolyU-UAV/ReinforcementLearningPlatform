import os
import sys
import datetime
import time
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as func

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from FlightAttitudeSimulator import Flight_Attitude_Simulator as fas
from algorithm.actor_critic.DDPG import DDPG
from utils.functions import *
from utils.classes import Normalization

timestep = 0
ENV = 'FlightAttitudeSimulator'
ALGORITHM = 'DDPG'
MAX_EPISODE = 1500
test_episode = []
test_reward = []
sumr_list = []


class Critic(nn.Module):
	def __init__(self, beta, state_dim, action_dim):
		super(Critic, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(self.state_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.action_value = nn.Linear(self.action_dim, 256)
		self.q = nn.Linear(256, 1)
		self.initialization()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		# self.device = 'cpu'
		self.to(self.device)

	def forward(self, s, a):
		sv = func.relu(self.fc1(s))
		sv = self.fc2(sv)

		av = func.relu(self.action_value(a))
		sav = func.relu(torch.add(sv, av))
		sav = self.q(sav)

		return sav

	def initialization(self):
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		nn.init.uniform_(self.fc1.bias.data, -f1, f1)

		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		nn.init.uniform_(self.fc2.bias.data, -f2, f2)

		f3 = 0.003
		nn.init.uniform_(self.q.weight.data, -f3, f3)
		nn.init.uniform_(self.q.bias.data, -f3, f3)


class Actor(nn.Module):
	def __init__(self, alpha, state_dim, action_dim):
		super(Actor, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.fc1 = nn.Linear(self.state_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.mu = nn.Linear(256, self.action_dim)
		self.initialization()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		# self.device = 'cpu'
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


def fullFillReplayMemory_with_Optimal(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
	print('Retraining...')
	print('Collecting...')
	fullFillCount = int(fullFillRatio * agent.memory.mem_size)
	fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
	_new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
	while agent.memory.mem_counter < fullFillCount:
		env.reset_random() if randomEnv else env.reset()
		_new_state.clear()
		_new_action.clear()
		_new_reward.clear()
		_new_state_.clear()
		_new_done.clear()
		while not env.is_terminal:
			if agent.memory.mem_counter % 100 == 0:
				print('replay_count = ', agent.memory.mem_counter)
			env.current_state = env.next_state.copy()  # 状态更新
			_action_from_actor = agent.choose_action(env.current_state, is_optimal=True)
			_action = agent.action_linear_trans(_action_from_actor)
			env.step_update(_action)
			# env.visualization()
			if is_only_success:
				_new_state.append(env.current_state)
				_new_action.append(env.current_action)
				_new_reward.append(env.reward)
				_new_state_.append(env.next_state)
				_new_done.append(1.0 if env.is_terminal else 0.0)
			else:
				agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
		if is_only_success:
			if env.terminal_flag == 3:
				print('Update Replay Memory......')
				agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
	"""
	:param randomEnv:       init env randomly
	:param fullFillRatio:   the ratio
	:return:                None
	"""
	print('Collecting...')
	fullFillCount = int(fullFillRatio * agent.memory.mem_size)
	fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
	while agent.memory.mem_counter < fullFillCount:
		env.reset_random() if randomEnv else env.reset()
		while not env.is_terminal:
			if agent.memory.mem_counter % 100 == 0:
				print('replay_count = ', agent.memory.mem_counter)
			env.current_state = env.next_state.copy()  # 状态更新
			# _action_from_actor = agent.choose_action(env.current_state, False, sigma=0.5)
			_action_from_actor = agent.choose_action_random()
			_action = agent.action_linear_trans(_action_from_actor)
			env.step_update(_action)
			# env.visualization()
			# if env.reward > 0:
			agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)


if __name__ == '__main__':
	log_dir = os.path.dirname(os.path.abspath(__file__)) + '/datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)

	RETRAIN = False

	env = fas(deg2rad(0))
	reward_norm = Normalization(shape=1)

	actor = Actor(1e-4, env.state_dim, env.action_dim)
	target_actor = Actor(1e-4, env.state_dim, env.action_dim)
	critic = Critic(3e-4, env.state_dim, env.action_dim)
	target_critic = Critic(3e-4, env.state_dim, env.action_dim)
	agent = DDPG(env=env,
				 gamma=0.99,
				 actor_soft_update=0.005,
				 critic_soft_update=0.005,
				 memory_capacity=20000,  # 10000
				 batch_size=512,
				 actor=actor,
				 target_actor=target_actor,
				 critic=critic,
				 target_critic=target_critic)
	agent.DDPG_info()

	if RETRAIN:
		print('RELOADING......')
		optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/net/'
		agent.actor.load_state_dict(torch.load(optPath + 'actor'))
		agent.critic.load_state_dict(torch.load(optPath + 'critic'))
		agent.target_actor.load_state_dict(torch.load(optPath + 'target_actor'))
		agent.target_critic.load_state_dict(torch.load(optPath + 'target_critic'))
		# agent.critic.init(True)
		# agent.target_critic.init(True)
		fullFillReplayMemory_with_Optimal(randomEnv=True, fullFillRatio=0.5, is_only_success=False)
	else:
		'''fullFillReplayMemory_Random'''
		fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5)
		'''fullFillReplayMemory_Random'''

	print('Start to train...')
	new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
	step = 0
	is_storage_only_success = False
	sigma0 = 0.5
	while agent.episode <= MAX_EPISODE:
		# env.reset()
		env.reset_random()
		sumr = 0
		new_state.clear()
		new_action.clear()
		new_reward.clear()
		new_state_.clear()
		new_done.clear()
		while not env.is_terminal:
			c = cv.waitKey(1)
			env.current_state = env.next_state.copy()
			if np.random.uniform(0, 1) < 0.00:
				action_from_actor = agent.choose_action_random()  # 有一定探索概率完全随机探索
			else:
				sigma = sigma0 - agent.episode * (sigma0 - 0.1) / MAX_EPISODE
				action_from_actor = agent.choose_action(env.current_state, False, sigma=sigma)  # 剩下的是神经网络加噪声
			action = agent.action_linear_trans(action_from_actor)
			env.step_update(action)
			step += 1
			if agent.episode % 10 == 0:
				env.visualization()
			sumr = sumr + env.reward
			if is_storage_only_success:
				new_state.append(env.current_state)
				new_action.append(env.current_action)
				new_reward.append(env.reward)
				new_state_.append(env.next_state)
				new_done.append(1.0 if env.is_terminal else 0.0)
			else:
				# if env.reward > 0:
				agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
			agent.learn(is_reward_ascent=False, iter=5)
		'''跳出循环代表回合结束'''
		if is_storage_only_success:
			if env.terminal_flag == 3:
				print('Update Replay Memory......')
				agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
		'''跳出循环代表回合结束'''
		print('Episode:', agent.episode, 'Cumulative reward:', round(sumr, 3))
		agent.episode += 1
		if agent.episode % 30 == 0:
			temp = simulationPath + 'trainNum_{}/'.format(agent.episode)
			os.mkdir(temp)
			time.sleep(0.01)
			print('Save net', agent.episode)
			agent.save_ac(msg='', path=temp)
