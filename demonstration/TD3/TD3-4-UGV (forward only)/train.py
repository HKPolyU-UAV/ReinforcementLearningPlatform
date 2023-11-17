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

from UGV import UGV as env
from algorithm.actor_critic.Twin_Delayed_DDPG import Twin_Delayed_DDPG as TD3
from utils.functions import *
from utils.classes import Normalization

timestep = 0
ENV = 'UGV'
ALGORITHM = 'TD3'
MAX_EPISODE = 30000
USE_R_NORM = False

r_norm = Normalization(shape=1)


class TD3Critic(nn.Module):
	def __init__(self, state_dim: int = 3, action_dim: int = 3, lr: float = 1e-4):
		super(TD3Critic, self).__init__()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.fc_q1_1 = nn.Linear(state_dim + action_dim, 256)
		self.fc_q1_2 = nn.Linear(256, 256)
		self.fc_q1_3 = nn.Linear(256, 1)

		self.fc_q2_1 = nn.Linear(state_dim + action_dim, 256)
		self.fc_q2_2 = nn.Linear(256, 256)
		self.fc_q2_3 = nn.Linear(256, 1)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		self.to(self.device)

	def forward(self, s, a):
		sav = torch.cat([s, a], 1)
		q1 = func.relu(self.fc_q1_1(sav))
		q1 = func.relu(self.fc_q1_2(q1))
		q1 = self.fc_q1_3(q1)

		q2 = func.relu(self.fc_q2_1(sav))
		q2 = func.relu(self.fc_q2_2(q2))
		q2 = self.fc_q2_3(q2)
		return q1, q2

	def q1(self, s, a):
		sav = torch.cat([s, a], 1)
		q1 = func.relu(self.fc_q1_1(sav))
		q1 = func.relu(self.fc_q1_2(q1))
		q1 = self.fc_q1_3(q1)
		return q1

	def q2(self, s, a):
		sav = torch.cat([s, a], 1)
		q2 = func.relu(self.fc_q2_1(sav))
		q2 = func.relu(self.fc_q2_2(q2))
		q2 = self.fc_q2_3(q2)
		return q2


class Actor(nn.Module):
	def __init__(self, alpha, state_dim, action_dim, a_min, a_max):
		super(Actor, self).__init__()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		# self.device = 'cpu'
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.a_min = torch.tensor(a_min, dtype=torch.float).to(self.device)
		self.a_max = torch.tensor(a_max, dtype=torch.float).to(self.device)
		self.off = (self.a_min + self.a_max) / 2.0
		self.gain = self.a_max - self.off
		# print(self.gain, self.off)
		self.fc1 = nn.Linear(self.state_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.mu = nn.Linear(256, self.action_dim)
		# self.initialization()
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


def fullFillReplayMemory_with_Optimal(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
	print('Retraining...')
	print('Collecting...')
	fullFillCount = int(fullFillRatio * agent.memory.mem_size)
	fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
	_new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
	while agent.memory.mem_counter < fullFillCount:
		env.reset(randomEnv)
		_new_state.clear()
		_new_action.clear()
		_new_reward.clear()
		_new_state_.clear()
		_new_done.clear()
		while not env.is_terminal:
			if agent.memory.mem_counter % 100 == 0:
				print('replay_count = ', agent.memory.mem_counter)
			env.current_state = env.next_state.copy()  # 状态更新
			_action = agent.choose_action(env.current_state, is_optimal=True, sigma=np.zeros(env.action_dim))
			env.step_update(_action)
			# env.visualization()
			r = r_norm(env.reward) if USE_R_NORM else env.reward
			if is_only_success:
				_new_state.append(env.current_state)
				_new_action.append(env.current_action)
				_new_reward.append(r)
				_new_state_.append(env.next_state)
				_new_done.append(1.0 if env.is_terminal else 0.0)
			else:
				agent.memory.store_transition(env.current_state, env.current_action, r, env.next_state, 1 if env.is_terminal else 0)
		if is_only_success:
			if env.terminal_flag == 3:
				print('Update Replay Memory......')
				agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
	print('Collecting...')
	fullFillCount = int(fullFillRatio * agent.memory.mem_size)
	fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
	while agent.memory.mem_counter < fullFillCount:
		env.reset(randomEnv)
		while not env.is_terminal:
			if agent.memory.mem_counter % 1000 == 0:
				print('replay_count = ', agent.memory.mem_counter)
			env.current_state = env.next_state.copy()  # 状态更新
			_action = agent.choose_action_random()
			env.step_update(_action)
			# env.visualization()
			# if env.reward > 0:
			r = r_norm(env.reward) if USE_R_NORM else env.reward
			agent.memory.store_transition(env.current_state, env.current_action, r, env.next_state, 1 if env.is_terminal else 0)


if __name__ == '__main__':
	log_dir = os.path.dirname(os.path.abspath(__file__)) + '/datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)

	RETRAIN = False

	env = env()
	reward_norm = Normalization(shape=1)

	actor = Actor(1e-4, env.state_dim, env.action_dim, env.action_range[:, 0], env.action_range[:, 1])
	target_actor = Actor(1e-4, env.state_dim, env.action_dim, env.action_range[:, 0], env.action_range[:, 1])
	critic = TD3Critic(env.state_dim, env.action_dim, lr=3e-4)
	target_critic = TD3Critic(env.state_dim, env.action_dim, lr=3e-4)
	env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'action_range': env.action_range, 'name': ENV}
	agent = TD3(env_msg=env_msg,
				gamma=0.99,
				noise_clip=0.5, noise_policy=0.2, policy_delay=2,
				actor_tau=0.005, td3critic_tau=0.005,
				memory_capacity=40000,
				batch_size=512,
				actor=actor,
				target_actor=target_actor,
				td3critic=critic,
				target_td3critic=target_critic)
	agent.TD3_info()

	if RETRAIN:
		print('RELOADING......')
		optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/net/'
		agent.actor.load_state_dict(torch.load(optPath + 'actor'))
		agent.target_actor.load_state_dict(torch.load(optPath + 'target_actor'))
		agent.td3critic.load_state_dict(torch.load(optPath + 'td3critic'))
		agent.target_td3critic.load_state_dict(torch.load(optPath + 'target_td3critic'))
		# agent.td3critic.init(True)
		# agent.target_td3critic.init(True)
		fullFillReplayMemory_with_Optimal(randomEnv=True, fullFillRatio=0.5, is_only_success=False)
	else:
		'''fullFillReplayMemory_Random'''
		fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.8)
		'''fullFillReplayMemory_Random'''

	print('Start to train...')
	new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
	step = 0
	is_storage_only_success = False
	sigma0 = (env.action_range[:, 1] - env.action_range[:, 0]) / 2 / 3
	while agent.episode <= MAX_EPISODE:
		# env.reset()
		env.reset(random=True)
		sumr = 0
		new_state.clear()
		new_action.clear()
		new_reward.clear()
		new_state_.clear()
		new_done.clear()
		while not env.is_terminal:
			env.current_state = env.next_state.copy()
			if np.random.uniform(0, 1) < 0.00:
				action = agent.choose_action_random()  # 有一定探索概率完全随机探索
			else:
				# sigma = sigma0 - agent.episode * (sigma0 - 0.1) / MAX_EPISODE
				sigma = sigma0
				action = agent.choose_action(env.current_state, is_optimal=False, sigma=sigma)
			env.step_update(action)
			step += 1
			if agent.episode % 10 == 0:
				env.visualization()
			sumr = sumr + env.reward
			r = r_norm(env.reward) if USE_R_NORM else env.reward
			if is_storage_only_success:
				new_state.append(env.current_state)
				new_action.append(env.current_action)
				new_reward.append(r)
				new_state_.append(env.next_state)
				new_done.append(1.0 if env.is_terminal else 0.0)
			else:
				# if env.reward > 0:
				agent.memory.store_transition(env.current_state, env.current_action, r, env.next_state, 1 if env.is_terminal else 0)
			agent.learn(is_reward_ascent=False, critic_random=False, iter=1)
		'''跳出循环代表回合结束'''
		if is_storage_only_success:
			if env.terminal_flag == 3:
				print('Update Replay Memory......')
				agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
		'''跳出循环代表回合结束'''
		print('Episode:', agent.episode, 'Cumulative reward:', round(sumr, 3))
		agent.episode += 1
		if agent.episode % 10 == 0:
			temp = simulationPath + 'trainNum_{}/'.format(agent.episode)
			os.mkdir(temp)
			time.sleep(0.01)
			print('Save net', agent.episode)
			agent.save_ac(msg='', path=temp)
