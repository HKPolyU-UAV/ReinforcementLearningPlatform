import torch
from utils.functions import *
from utils.classes import Actor, Critic, ReplayBuffer, GaussianNoise
import torch.nn.functional as func

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class DDPG:
	def __init__(self,
				 env,
				 gamma: float = 0.99,
				 actor_soft_update: float = 1e-2,
				 critic_soft_update: float = 1e-2,
				 memory_capacity: int = 5000,
				 batch_size: int = 512,
				 actor: Actor = Actor(),
				 target_actor: Actor = Actor(),
				 critic: Critic = Critic(),
				 target_critic: Critic = Critic()):
		self.env = env

		'''DDPG'''
		self.gamma = gamma
		self.actor_tau = actor_soft_update
		self.critic_tau = critic_soft_update
		self.memory = ReplayBuffer(memory_capacity, batch_size, self.env.state_dim, self.env.action_dim)
		'''DDPG'''

		'''network'''
		self.actor = actor
		self.target_actor = target_actor
		self.critic = critic
		self.target_critic = target_critic
		'''network'''

		self.noise_gaussian = GaussianNoise(mu=np.zeros(self.env.action_dim))
		self.update_network_parameters()

		self.episode = 0
		self.reward = 0

	def choose_action_random(self):
		"""
        :brief:     因为该函数与choose_action并列
        :return:    random action
        """
		return np.random.uniform(low=-1, high=1, size=self.env.action_dim)

	def choose_action(self, state, is_optimal=False, sigma=1 / 3):
		self.actor.eval()  # 切换到测试模式
		t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)  # get the tensor of the state
		mu = self.actor(t_state).to(self.actor.device)  # choose action
		if is_optimal:
			mu_prime = mu
		else:
			mu_prime = mu + torch.tensor(self.noise_gaussian(sigma=sigma), dtype=torch.float).to(self.actor.device)  # action with gaussian noise
			# mu_prime = mu + torch.tensor(self.noise_OU(), dtype=torch.float).to(self.actor.device)             # action with OU noise
		self.actor.train()  # 切换回训练模式
		mu_prime_np = mu_prime.cpu().detach().numpy()
		return np.clip(mu_prime_np, -1, 1)  # 将数据截断在[-1, 1]之间

	def evaluate(self, state):
		self.target_actor.eval()
		t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)  # get the tensor of the state
		act = self.target_actor(t_state).to(self.target_actor.device)
		return act.cpu().detach().numpy()

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
		print('agent name：', self.env.name)
		print('state_dim:', self.env.state_dim)
		print('action_dim:', self.env.action_dim)
		print('action_range:', self.env.action_range)

	def action_linear_trans(self, action):
		# the action output
		linear_action = []
		for i in range(self.env.action_dim):
			a = min(max(action[i], -1), 1)
			maxa = self.env.action_range[i][1]
			mina = self.env.action_range[i][0]
			k = (maxa - mina) / 2
			b = (maxa + mina) / 2
			linear_action.append(k * a + b)
		return np.array(linear_action)
