from common.common_cls import *
import cv2 as cv


class Proximal_Policy_Optimization_Discrete:
	def __init__(self,
				 env,
				 gamma: float = 0.99,
				 K_epochs: int = 10,
				 eps_clip: float = 0.2,
				 buffer_size: int = 1200,
				 actor: SoftmaxActor = SoftmaxActor(),
				 critic: Critic = Critic(),
				 path: str = ''):
		self.env = env
		self.gamma = gamma

		'''PPO'''
		self.gamma = gamma  # discount factor
		self.K_epochs = K_epochs  # 每隔 timestep_num 学习一次
		self.eps_clip = eps_clip
		self.path = path
		self.buffer = RolloutBuffer(buffer_size, self.env.state_dim, self.env.action_dim)
		self.buffer2 = RolloutBuffer2(self.env.state_dim, self.env.action_dim)
		self.actor = actor
		self.critic = critic
		'''PPO'''

		self.device = 'cpu'
		self.episode = 0

	def choose_action_random(self):
		"""
		:brief:     因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
		:return:    random action
		"""
		_a = []
		for _num in self.env.action_num:
			_a.append(np.random.choice(_num))
		return np.array(_a)

	def choose_action(self, state):
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
			_a, _a_log_prob, _ = self.actor.choose_action(t_state)
			_s_value = self.critic(t_state).cpu()
		return _a, t_state, _a_log_prob, _s_value

	def evaluate(self, state):
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
			_a = self.actor.evaluate(t_state)
		return _a

	def train_evaluate(self, state, action):
		_dist = Categorical(probs=self.actor.forward(state))

		if self.env.action_dim == 1:
			action = action.reshape(-1, self.env.action_dim)

		_log_probs = torch.mean(_dist.log_prob(action), dim=1)
		_dist_entropy = torch.mean(_dist.entropy(), dim=1)
		_s_values = self.critic(state)
		return _log_probs, _s_values, _dist_entropy

	def agent_evaluate(self, test_num):
		r = 0
		for _ in range(test_num):
			self.env.reset_random()
			while not self.env.is_terminal:
				self.env.current_state = self.env.next_state.copy()
				_action_from_actor = self.evaluate(self.env.current_state)
				_action = self.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将动作转换到实际范围上
				self.env.step_update(_action)  # 环境更新的action需要是物理的action
				r += self.env.reward
				self.env.show_dynamic_image(isWait=False)  # 画图
		cv.destroyAllWindows()
		r /= test_num
		return r

	def action_linear_trans(self, action):
		"""
		@param action:
		@return:
		"""
		linear_action = []
		for _a, _action_space in zip(action, self.env.action_space):
			linear_action.append(_action_space[_a])
		return np.array(linear_action)

	def learn(self, adv_norm=False):
		"""
		@return:
		"""
		'''1. 计算轨迹中每一个状态的累计回报'''
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + self.gamma * discounted_reward
			rewards.insert(0, discounted_reward)

		'''2. 奖励归一化'''
		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		'''3. 将 numpy 数据转化为 tensor'''
		with torch.no_grad():
			old_states = torch.FloatTensor(self.buffer.states).detach().to(self.device)
			old_actions = torch.FloatTensor(self.buffer.actions).detach().to(self.device)
			old_log_probs = torch.FloatTensor(self.buffer.log_probs).detach().to(self.device)
			old_state_values = torch.FloatTensor(self.buffer.state_values).detach().to(self.device)

		'''4. 计算优势函数'''
		advantages = rewards.detach() - old_state_values.detach()
		if adv_norm:
			advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

		'''5. 学习 K 次'''
		for _ in range(self.K_epochs):
			'''5.1 Evaluating old actions and values'''
			log_probs, state_values, dist_entropy = self.train_evaluate(old_states, old_actions)

			'''5.2 match state_values tensor dimensions with rewards tensor'''
			state_values = torch.squeeze(state_values)

			'''5.3 Finding the ratio (pi_theta / pi_theta__old)'''
			ratios = torch.exp(log_probs - old_log_probs.detach())

			'''5.4 Finding Surrogate Loss'''
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

			'''5.5 final loss of clipped objective PPO'''
			actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
			self.actor.optimizer.zero_grad()
			actor_loss.mean().backward()
			self.actor.optimizer.step()
			# loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy

			critic_loss = 0.5 * func.mse_loss(state_values, rewards)
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()

	def save_models(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()

	def save_models_all(self):
		self.actor.save_all_net()
		self.critic.save_all_net()

	def load_models(self, path):
		"""
		:brief:         only for test
		:param path:    file path
		:return:
		"""
		print('...loading checkpoint...')
		self.actor.load_state_dict(torch.load(path + 'PPO_Actor'))
		self.critic.load_state_dict(torch.load(path + 'PPO_Critic'))

	def PPO_info(self):
		print('agent name：', self.env.name)
		print('state_dim:', self.env.state_dim)
		print('action_dim:', self.env.action_dim)
		print('action num:', self.env.action_num)
		print('action_range:', self.env.action_space)
