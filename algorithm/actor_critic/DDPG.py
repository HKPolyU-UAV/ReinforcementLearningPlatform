import numpy as np
import torch
from utils.functions import *
from utils.classes import Actor, Critic, ReplayBuffer, GaussianNoise
import torch.nn.functional as func

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class DDPG:
	def __init__(self,
				 env_msg:dict,
				 gamma: float = 0.99,
				 actor_soft_update: float = 1e-2,
				 critic_soft_update: float = 1e-2,
				 memory_capacity: int = 5000,
				 batch_size: int = 512,
				 actor: Actor = Actor(),
				 target_actor: Actor = Actor(),
				 critic: Critic = Critic(),
				 target_critic: Critic = Critic()):

		'''DDPG'''
		self.env_msg = env_msg
		self.gamma = gamma
		self.actor_tau = actor_soft_update
		self.critic_tau = critic_soft_update
		self.memory = ReplayBuffer(memory_capacity, batch_size, env_msg['state_dim'], env_msg['action_dim'])
		'''DDPG'''

		'''network'''
		self.actor = actor
		self.target_actor = target_actor
		self.target_actor.load_state_dict(self.actor.state_dict())

		self.critic = critic
		self.target_critic = target_critic
		self.target_critic.load_state_dict(self.critic.state_dict())
		'''network'''

		self.a_min = env_msg['action_range'][:, 0]
		self.a_max = env_msg['action_range'][:, 1]

		self.episode = 0
		self.reward = 0

	def choose_action_random(self):
		"""
        :brief:     因为该函数与choose_action并列
        :return:    random action
        """
		return np.random.uniform(low=self.a_min, high=self.a_max)

	def choose_action(self, state, is_optimal=False, sigma: float = 1 / 3, action_dim: int = 1):
		t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
		mu = self.actor(t_state).to(self.actor.device)
		if not is_optimal:
			noise = torch.Tensor(np.random.normal(0, sigma, size=action_dim)).to(self.actor.device)
			mu = mu + noise
		mu_np = mu.cpu().detach().numpy().flatten()
		mu_np = np.clip(mu_np, self.a_min, self.a_max)
		return mu_np

	def evaluate(self, state):
		t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
		return self.target_actor(t_state).to(self.target_actor.device).cpu().detach().numpy().flatten()

	def learn(self, is_reward_ascent=True, iter=1):
		if self.memory.mem_counter < self.memory.batch_size:
			return
		for _ in range(iter):
			s, a, r, s_, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
			s = torch.tensor(s, dtype=torch.float).to(self.critic.device)
			a = torch.tensor(a, dtype=torch.float).to(self.critic.device)
			r = torch.tensor(r, dtype=torch.float).to(self.critic.device)
			s_ = torch.tensor(s_, dtype=torch.float).to(self.critic.device)
			done = torch.tensor(done, dtype=torch.float).to(self.critic.device)

			with torch.no_grad():
				Q_ = self.target_critic(s_, self.target_actor(s_))
				target_Q = r + self.gamma * (1 - done) * Q_

			current_Q = self.critic(s, a)
			critic_loss = func.mse_loss(target_Q, current_Q)
			# Optimize the critic
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()

			# Freeze critic networks so you don't waste computational effort
			for params in self.critic.parameters():
				params.requires_grad = False

			# Compute the actor loss
			actor_loss = -self.critic(s, self.actor(s)).mean()
			# Optimize the actor
			self.actor.optimizer.zero_grad()
			actor_loss.backward()
			self.actor.optimizer.step()

			# Unfreeze critic networks
			for params in self.critic.parameters():
				params.requires_grad = True

			self.update_network_parameters()

	def update_network_parameters(self):
		"""
        :return:        None
        """
		for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.critic_tau) + param.data * self.critic_tau)  # soft update
		for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.actor_tau) + param.data * self.actor_tau)  # soft update

	def save_ac(self, msg, path):
		torch.save(self.actor.state_dict(), path + 'actor' + msg)
		torch.save(self.target_actor.state_dict(), path + 'target_actor' + msg)
		torch.save(self.critic.state_dict(), path + 'critic' + msg)
		torch.save(self.target_critic.state_dict(), path + 'target_critic' + msg)

	def DDPG_info(self):
		print('agent name：', self.env_msg['name'])
		print('state_dim:', self.env_msg['state_dim'])
		print('action_dim:', self.env_msg['action_dim'])
		print('action_range:', self.env_msg['action_range'])
