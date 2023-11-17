import numpy as np
import torch
from utils.functions import *
from utils.classes import Actor, TD3Critic, ReplayBuffer
import torch.nn.functional as func
from torch.distributions import Normal

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Twin_Delayed_DDPG:
	def __init__(self,
				 env_msg: dict,
				 gamma: float = 0.9,
				 noise_clip: float = 1 / 2,
				 noise_policy: float = 1 / 4,
				 policy_delay: int = 5,
				 td3critic_tau: float = 5e-3,
				 actor_tau: float = 5e-3,
				 memory_capacity: int = 5000,
				 batch_size: int = 64,
				 actor: Actor = Actor(),
				 target_actor: Actor = Actor(),
				 td3critic: TD3Critic = TD3Critic(),
				 target_td3critic: TD3Critic = TD3Critic()):

		'''Twin-Delay-DDPG'''
		self.gamma = gamma
		self.env_msg = env_msg
		self.policy_delay = policy_delay
		self.policy_delay_iter = 0
		self.td3critic_tau = td3critic_tau
		self.actor_tau = actor_tau
		self.memory = ReplayBuffer(memory_capacity, batch_size, env_msg['state_dim'], env_msg['action_dim'])
		'''Twin-Delay-DDPG'''

		'''network'''
		self.actor = actor
		self.target_actor = target_actor
		self.td3critic = td3critic
		self.target_td3critic = target_td3critic
		self.device = self.td3critic.device
		self.actor_replace_iter = 0
		'''network'''

		self.a_min = torch.FloatTensor(env_msg['action_range'][:, 0])
		self.a_max = torch.FloatTensor(env_msg['action_range'][:, 1])
		self.noise_clip = torch.FloatTensor(noise_clip * (self.a_max - self.a_min) / 2)
		# self.noise_policy = torch.FloatTensor(noise_policy * (self.a_max - self.a_min) / 2)
		self.noise_policy = Normal(torch.zeros(env_msg['action_dim']), noise_clip * (self.a_max - self.a_min) / 2)

		self.episode = 0
		self.reward = 0

	def choose_action_random(self):
		a = torch.rand(self.env_msg['action_dim']) * (self.a_max - self.a_min) + self.a_min
		return a.cpu().detach().numpy().flatten()

	def choose_action(self, state, is_optimal, sigma: np.ndarray):
		t_state = torch.FloatTensor(state) if self.device == 'cpu' else torch.cuda.FloatTensor(state)
		mu = self.actor(t_state).cpu().detach().numpy().flatten()
		if not is_optimal:
			noise = np.random.multivariate_normal(np.zeros_like(sigma), np.diag(sigma ** 2))	# mu sigma^2
			mu = mu + noise
		return np.clip(mu, self.a_min.cpu().detach().numpy().flatten(), self.a_max.cpu().detach().numpy().flatten())

	def evaluate(self, state):
		t_state = torch.tensor(state, dtype=torch.float).to(self.device)
		act = self.target_actor(t_state).to(self.device)
		return act.cpu().detach().numpy().flatten()

	def learn(self, is_reward_ascent=False, critic_random=True, iter=1):
		if self.memory.mem_counter < self.memory.batch_size:
			return
		for _ in range(iter):
			s, a, r, s_, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
			s = torch.FloatTensor(s).to(self.device)
			a = torch.FloatTensor(a).to(self.device)
			r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
			s_ = torch.FloatTensor(s_).to(self.device)
			done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

			'''
				Shapes of tensors:
				s: [batch_size, state_dim]
				a: [batch_size, action_dim]
				r: [batch_size, 1]
				s_: [batch_size, state_dim]
				done: [batch_size, 1]
			'''
			with torch.no_grad():
				noise = self.noise_policy.sample((self.memory.batch_size, 1)).squeeze()
				noise = torch.maximum(torch.minimum(noise, self.noise_clip), -self.noise_clip)
				next_action = self.target_actor(s_).cpu() + noise
				next_action = torch.maximum(torch.minimum(next_action, self.a_max), self.a_min)

				target_Q1, target_Q2 = self.target_td3critic(s_, next_action.to(self.device))
				target_Q = r + self.gamma * done * torch.min(target_Q1, target_Q2)

			# 得到当前的 Q value
			current_Q1, current_Q2 = self.td3critic(s, a)

			# Compute the critic loss
			critic_loss = func.mse_loss(current_Q1, target_Q) + func.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.td3critic.optimizer.zero_grad()
			critic_loss.backward()
			self.td3critic.optimizer.step()

			self.policy_delay_iter += 1

			'''延迟更新'''
			if self.policy_delay_iter % self.policy_delay == 0:
				for params in self.td3critic.parameters():
					params.requires_grad = False

				mu = self.actor.forward(s)
				if critic_random and np.random.randint(1, 2) == 2:
					actor_loss = -self.td3critic.q2(s, mu).mean()
				else:
					actor_loss = -self.td3critic.q1(s, mu).mean()
				self.actor.optimizer.zero_grad()
				actor_loss.backward()
				self.actor.optimizer.step()

				for params in self.td3critic.parameters():
					params.requires_grad = True

			self.update_network_parameters()

	def update_network_parameters(self, is_target_critic_delay: bool = False):
		"""
		:return:        None
		"""
		if not is_target_critic_delay:
			for target_param, param in zip(self.target_td3critic.parameters(), self.td3critic.parameters()):
				target_param.data.copy_(
					target_param.data * (1.0 - self.td3critic_tau) + param.data * self.td3critic_tau)

		if self.policy_delay_iter % self.policy_delay == 0:
			for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
				target_param.data.copy_(
					target_param.data * (1.0 - self.actor_tau) + param.data * self.actor_tau)
			if is_target_critic_delay:
				for target_param, param in zip(self.target_td3critic.parameters(), self.td3critic.parameters()):
					target_param.data.copy_(
						target_param.data * (1.0 - self.td3critic_tau) + param.data * self.td3critic_tau)

	def save_ac(self, msg, path):
		torch.save(self.actor.state_dict(), path + 'actor' + msg)
		torch.save(self.target_actor.state_dict(), path + 'target_actor' + msg)
		torch.save(self.td3critic.state_dict(), path + 'td3critic' + msg)
		torch.save(self.target_td3critic.state_dict(), path + 'target_td3critic' + msg)

	def TD3_info(self):
		print('agent name：', self.env_msg['name'])
		print('state_dim:', self.env_msg['state_dim'])
		print('action_dim:', self.env_msg['action_dim'])
		print('action_range:', self.env_msg['action_range'])
