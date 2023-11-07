import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F

from utils.classes import PPOActor_Gaussian, PPOCritic, RolloutBuffer, RolloutBuffer2

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Proximal_Policy_Optimization2:
	def __init__(self,
				 env_msg: dict = None,
				 ppo_msg: dict = None,
				 actor: PPOActor_Gaussian = PPOActor_Gaussian(),
				 critic: PPOCritic = PPOCritic()):
		self.env_msg = env_msg
		self.ppo_msg = ppo_msg
		'''PPO'''
		self.gamma = ppo_msg['gamma']
		self.K_epochs = ppo_msg['K_epochs']
		self.eps_clip = ppo_msg['eps_clip']
		self.buffer = RolloutBuffer(ppo_msg['buffer_size'], self.env_msg['state_dim'], self.env_msg['action_dim'])
		self.buffer2 = RolloutBuffer2(self.env_msg['state_dim'], self.env_msg['action_dim'])
		self.actor_lr = ppo_msg['a_lr']
		self.critic_lr = ppo_msg['c_lr']
		'''PPO'''

		'''Trick params'''
		self.set_adam_eps = ppo_msg['set_adam_eps']
		self.lmd = ppo_msg['lmd']
		self.use_adv_norm = ppo_msg['use_adv_norm']
		self.mini_batch_size = ppo_msg['mini_batch_size']
		self.entropy_coef = ppo_msg['entropy_coef']
		self.use_grad_clip = ppo_msg['use_grad_clip']
		self.use_lr_decay = ppo_msg['use_lr_decay']
		self.max_train_steps = ppo_msg['max_train_steps']
		self.using_mini_batch = ppo_msg['using_mini_batch']
		'''Trick params'''

		'''networks'''
		self.actor = actor
		self.critic = critic
		if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
			self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
			self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)
		else:
			self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
			self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

		self.loss = nn.MSELoss()
		self.device = device  # 建议使用 CPU 训练
		'''networks'''

		self.cnt = 0

	def evaluate(self, state):
		with torch.no_grad():
			t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
			action_mean = self.actor(t_state)
		return action_mean.detach().cpu().numpy().flatten()

	def choose_action(self, state: np.ndarray):
		with torch.no_grad():
			t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
			dist = self.actor.get_dist(t_state)
			a = dist.sample()  # Sample the action according to the probability distribution
			a = torch.maximum(torch.minimum(a, self.actor.a_max), self.actor.a_min)
			a_logprob = dist.log_prob(a)  # The log probability density of the action
		return a.detach().cpu().numpy().flatten(), a_logprob.detach().cpu().numpy().flatten()

	def learn(self, current_steps, buf_num: int = 1):
		"""
		@note: 	 network update
		@return: None
		"""
		'''前期数据处理'''
		if buf_num == 1:
			s, a, a_lp, r, s_, done, success = self.buffer.to_tensor()
		else:
			s, a, a_lp, r, s_, done, success = self.buffer2.to_tensor()
		adv = []
		gae = 0.
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_)
			deltas = r + self.gamma * (1.0 - success) * vs_ - vs
			for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
				gae = delta + self.gamma * self.lmd * gae * (1.0 - d)
				adv.insert(0, gae)
			adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
			v_target = adv + vs
			if self.use_adv_norm:  # Trick 1:advantage normalization
				adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

		if self.using_mini_batch:
			for _ in range(self.K_epochs):  # 每次的轨迹数据学习 K_epochs 次
				for index in BatchSampler(SubsetRandomSampler(range(self.buffer.batch_size)), self.mini_batch_size, False):
					dist_now = self.actor.get_dist(s[index])
					dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
					a_logprob_now = dist_now.log_prob(a[index])

					# a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
					ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_lp[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

					surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
					surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv[index]
					actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
					# Update actor
					self.optimizer_actor.zero_grad()
					actor_loss.mean().backward()

					if self.use_grad_clip:  # Trick 7: Gradient clip
						torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
					self.optimizer_actor.step()

					v_s = self.critic(s[index])
					critic_loss = F.mse_loss(v_target[index], v_s)

					# Update critic
					self.optimizer_critic.zero_grad()
					critic_loss.backward()
					if self.use_grad_clip:  # Trick 7: Gradient clip
						torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
					self.optimizer_critic.step()
		else:
			for _ in range(self.K_epochs):  # 每次的轨迹数据学习 K_epochs 次
				dist_now = self.actor.get_dist(s)
				dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
				a_logprob_now = dist_now.log_prob(a)

				# a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
				ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_lp.sum(1, keepdim=True))

				surr1 = ratios * adv  # Only calculate the gradient of 'a_logprob_now' in ratios
				surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
				actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
				# Update actor
				self.optimizer_actor.zero_grad()
				actor_loss.mean().backward()

				if self.use_grad_clip:  # Trick 7: Gradient clip
					torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
				self.optimizer_actor.step()

				v_s = self.critic(s)
				critic_loss = F.mse_loss(v_target, v_s)

				# Update critic
				self.optimizer_critic.zero_grad()
				critic_loss.backward()
				if self.use_grad_clip:  # Trick 7: Gradient clip
					torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
				self.optimizer_critic.step()

		if self.use_lr_decay:  # Trick 6:learning rate Decay
			self.lr_decay(current_steps)

	def lr_decay(self, total_steps):
		if total_steps < self.max_train_steps:
			lr_a_now = self.actor_lr * (1 - total_steps / self.max_train_steps)
			lr_c_now = self.critic_lr * (1 - total_steps / self.max_train_steps)
			lr_a_now = max(lr_a_now, 1e-6)
			lr_c_now = max(lr_c_now, 1e-6)
			for p in self.optimizer_actor.param_groups:
				p['lr'] = lr_a_now
			for p in self.optimizer_critic.param_groups:
				p['lr'] = lr_c_now

	def action_linear_trans(self, action):
		# the action output
		linear_action = []
		for i in range(self.env_msg['action_dim']):
			a = min(max(action[i], -1), 1)
			maxa = self.env_msg['action_range'][i][1]
			mina = self.env_msg['action_range'][i][0]
			k = (maxa - mina) / 2
			b = (maxa + mina) / 2
			linear_action.append(k * a + b)
		return np.array(linear_action)

	def save_ac(self, msg, path):
		torch.save(self.actor.state_dict(), path + 'actor' + msg)
		torch.save(self.critic.state_dict(), path + 'critic' + msg)

	def PPO2_info(self):
		print('agent name：', self.env_msg['name'])
		print('state_dim:', self.env_msg['state_dim'])
		print('action_dim:', self.env_msg['action_dim'])
		print('action_range:', self.env_msg['action_range'])
