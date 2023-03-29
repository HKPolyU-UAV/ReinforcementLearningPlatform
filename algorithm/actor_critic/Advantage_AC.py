import pandas as pd
from common.common_cls import *
import cv2 as cv

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Advantage_AC:
	def __init__(self,
				 env,
				 gamma: float = 0.99,
				 timestep_num: int = 10,
				 actor: SoftmaxActor = SoftmaxActor(),
				 critic: Critic = Critic(),
				 path: str = ''):
		"""
		@param env:
		@param gamma:
		@param timestep_num:
		@param actor:
		@param critic:
		@param path:
		"""
		self.env = env

		'''A2C'''
		self.gamma = gamma  # discount factor
		self.timestep_num = timestep_num  # 每隔 timestep_num 学习一次
		self.path = path
		'''A2C'''

		'''networks'''
		self.actor = actor
		self.critic = critic
		self.device = device  # 建议使用 CPU 训练
		'''networks'''

		self.episode = 0
		self.reward = 0

		self.save_episode = []  # 保存的每一个回合的回合数
		self.save_reward = []  # 保存的每一个回合的奖励
		self.save_time = []
		self.save_step = []  # 保存的每一步的步数
		self.save_stepreward = []  # 保存的每一步的奖励

	def choose_action_random(self):
		"""
		@note:		随机选择每一维动作的索引，记住是索引
		@return:	random action
		"""
		_a = []
		for _num in self.env.action_num:
			_a.append(np.random.randint(low=0, high=_num))
		return np.array(_a)

	def choose_action(self, state, deterministic=False):
		t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
		p = torch.squeeze(self.actor(t_state).cpu().detach()).numpy()  # probability distribution(numpy), n 个分布列
		p = np.split(p, [np.sum(self.env.action_num[0:i]) for i in range(self.env.action_dim - 1)])
		_a = []
		_lg_prob = []
		if deterministic:
			for i in range(self.env.action_dim):
				index = np.argmax(p[i])
				_a.append(index)  # Select the action with the highest probability
				_lg_prob.append(np.log(p[i][index]))
			return np.array(_a), np.array(_lg_prob)
		else:
			for i in range(self.env.action_dim):
				index = np.random.choice(range(self.env.action_num[i]), p=p[i] / np.sum(p[i]))
				_a.append(index)
				_lg_prob.append(np.log(p[i][index]))
			return np.array(_a), np.array(_lg_prob)

	def evaluate(self, state):
		return self.choose_action(state, True)

	def agent_evaluate(self, test_num: int = 5):
		sum_r = 0
		for _ in range(test_num):
			self.env.reset_random()
			r = 0
			while not self.env.is_terminal:
				cv.waitKey(1)
				self.env.current_state = self.env.next_state.copy()
				action_from_actor, _ = self.evaluate(self.env.current_state)
				action = self.action_index_find(action_from_actor)
				self.env.step_update(action)  # 环境更新的action需要是物理的action
				r += self.env.reward
			sum_r += r
		cv.destroyAllWindows()
		return sum_r / test_num, self.env.time

	def learn(self, r: np.ndarray, lg_prob: np.ndarray, vs: np.ndarray):
		t_vs = torch.FloatTensor(vs).squeeze().to(self.device)  # 这里的a是索引号，必须是int
		t_lg_prob = torch.FloatTensor(lg_prob).to(self.device)
		R = 0
		traj_r = []
		for _r in r[::-1]:
			R = _r + self.gamma * R
			traj_r.insert(0, R)
		with torch.no_grad():
			traj_r = torch.FloatTensor(traj_r).squeeze().to(self.device)
			advantages = traj_r - t_vs

		# Update actor
		t_lg_prob.requires_grad_()
		t_vs.requires_grad_()
		actor_loss = -torch.sum(t_lg_prob * advantages)
		self.actor.optimizer.zero_grad()
		actor_loss.backward()
		self.actor.optimizer.step()

		# Update critic
		critic_loss = func.smooth_l1_loss(t_vs, traj_r)
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
		self.actor.load_state_dict(torch.load(path + 'Actor_A2C'))
		self.critic.load_state_dict(torch.load(path + 'Critic_A2C'))

	def load_actor_optimal(self, path, file):
		print('...loading optimal...')
		self.actor.load_state_dict(torch.load(path + file))

	def A2C_info(self):
		print('agent name:', self.env.name)
		print('state_dim:', self.env.state_dim)
		print('action_dim:', self.env.action_dim)
		print('action_num', self.env.action_num)
		print('action_space_nn', self.env.action_space)
		print('action_range:', self.env.action_range)

	def action_index_find(self, action):
		"""
		@param action:		the index of the action space
		@return:			real physical action
		"""
		_a = []
		for i in range(self.env.action_dim):
			_a.append(self.env.action_space[i][action[i]])
		return np.array(_a)

	def saveData_Step_Reward(self, step, reward, is2file=False, filename='StepReward.csv'):
		if is2file:
			data = pd.DataFrame({
				'step:': self.save_step,
				'stepreward': self.save_stepreward,
			})
			data.to_csv(self.path + filename, index=False, sep=',')
		else:
			self.save_step.append(step)
			self.save_stepreward.append(reward)

	def saveData_EpisodeReward(self, episode, time, reward, is2file=False, filename='EpisodeReward.csv'):
		if is2file:
			data = pd.DataFrame({
				'episode': self.save_episode,
				'time': self.save_time,
				'reward': self.save_reward
			})
			data.to_csv(self.path + filename, index=False, sep=',')
		else:
			self.save_episode.append(episode)
			self.save_time.append(time)
			self.save_reward.append(reward)
