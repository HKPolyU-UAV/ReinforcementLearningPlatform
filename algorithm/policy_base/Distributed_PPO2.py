import time
from common.common_cls import *
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import cv2 as cv

# import pandas as pd

"""
	Note: CPU is recommended for worker, and GPU is recommended for chief.
"""


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


class Worker(mp.Process):
	def __init__(self,
				 env,
				 name: str,
				 index: int,
				 total_collection: int,
				 ref_permit: np.ndarray,
				 share_permit: shared_memory.SharedMemory.buf,
				 ref_buffer: np.ndarray,
				 share_buffer: shared_memory.SharedMemory.buf,
				 policy_net_shape: list,
				 share_net_list: list,
				 policy: PPOActorCritic,
				 gamma: float,
				 action_std_decay_freq: int,
				 action_std_decay_rate: float,
				 min_action_std: float,
				 action_std_init: float):
		"""
		@param env:						RL 环境，用于交互产生数据
		@param name:					worker 名字
		@param index:					worker 编号
		@param total_collection:		一共采集数据次数
		@param ref_permit:				允许采集标志位 (共享)
		@param share_permit:			允许采集标志位共享内存对应的地址
		@param ref_buffer:				采集数据 (共享)
		@param share_buffer:			采集数据共享内存对应的地址
		@param policy_net_shape:		网络参数尺寸的 list
		@param share_net_list:			网络参数的共享内存对应的地址
		@param policy:					model
		@param gamma:					折扣因子 (用于计算一条轨迹中各个状态的累计奖励)
		@param action_std_decay_freq:	动作方差缩减的频率
		@param action_std_decay_rate:	动作方差缩减的速度
		@param min_action_std:			最小动作方差
		@param action_std_init:			初始动作方差
		"""
		super(Worker, self).__init__()
		setup_seed(3407)
		'''worker configuration'''
		self.name = name
		self.index = index
		'''worker configuration'''

		'''multi process & logic'''
		self.permit = np.ndarray(ref_permit.shape, ref_permit.dtype, share_permit)
		self.permit[0] = 1
		self.total_collection = total_collection  # 记录一共需要采集的次数
		self.collection = 0  # 记录已经采集的次数
		self.buffer = np.ndarray(ref_buffer.shape, ref_buffer.dtype, share_buffer)  # 存储数据
		self.buffer_size = self.buffer.shape[0]
		self.policy_net_shape = policy_net_shape
		self.net_param = []
		for _buf, _shape in zip(share_net_list, self.policy_net_shape):
			self.net_param.append(np.ndarray(_shape, np.float32, _buf))
		'''multi process & logic'''

		'''env and net'''
		self.env = env
		self.policy = policy
		self.device = 'cpu'
		self.policy.to(self.device)  # worker 中，policy 都在 cpu 上
		self.gamma = gamma
		self.action_std_decay_freq = action_std_decay_freq
		self.action_std_decay_rate = action_std_decay_rate
		self.min_action_std = min_action_std
		self.action_std = action_std_init
		self.timestep = 0
		self.load_net_params()
		'''env and net'''

		self.episode = 0

	def choose_action(self, state):
		"""
		@param state:	state of the env
		@return:		action (tensor), state (tensor), the log-probability of action (tensor), and state value (tensor)
		"""
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
			action, action_log_prob, state_val = self.policy.act(t_state)
		return action, t_state, action_log_prob, state_val

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
		return linear_action

	def set_action_std(self, new_action_std):
		self.action_std = new_action_std
		self.policy.set_action_std(new_action_std)

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		self.action_std = self.action_std - action_std_decay_rate
		self.action_std = round(self.action_std, 4)
		if self.action_std <= min_action_std:
			self.action_std = min_action_std
			print("setting actor output action_std to min_action_std : ", self.action_std)
		else:
			print("setting actor output action_std to : ", self.action_std)
		self.set_action_std(self.action_std)

	def package_rewards(self, buffer_r, buffer_done):
		"""
		@note:					r 和 done 需要对应起来
		@param buffer_r:		the immediate reward of trajectories
		@param buffer_done:		the done flag of trajectories
		@return:				the cumulative reward of trajectories
		"""
		# rewards = []
		# _l = len(buffer_r)
		# rewards = np.zeros(_l)
		discounted_reward = 0
		# 倒过来存储，计算值函数是从末端往回推
		for reward, is_terminal, _index in zip(reversed(buffer_r), reversed(buffer_done), reversed(range(self.buffer_size))):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + self.gamma * discounted_reward
			self.buffer[_index, -1] = discounted_reward

	def load_net_params(self):
		for _global_param, _local_param in zip(self.net_param, self.policy.parameters()):
			with torch.no_grad():
				_local_param.copy_(torch.tensor(_global_param))

	def run(self):
		"""
			@note:		DPPO2 中，worker 只负责产生数据，并不负责训练
					global_permit 为1时，开始收集数据，手机完成时，置 0。主进程检测到所有标志位都为0时，开始学习，学习完毕，将该标志位置 1。
					换言之，子进程只有将 global_permit 置 0 的权限；主进程只有将 global_permit 置 1 的权限
			@return:
		"""
		start_collection = 0
		while self.collection < self.total_collection:
			if self.permit[0] == 1:
				index = 0
				self.load_net_params()		# 将模型参数加载进来
				self.policy.to(self.device)  # 将模型放到 cpu 上
				# print('========= EPISODE START =========')
				# print('Collecting data...')
				_immediate_r = np.ones(self.buffer_size)  # 用于记录轨迹的 r
				_done = np.ones(self.buffer_size)  # 用于记录轨迹的 done
				sumr = 0
				while index < self.buffer_size:  # 如果数据还没有搜集满
					self.env.reset_random()
					while not self.env.is_terminal:
						'''测试共享内存'''
						# self.buffer[index, 0: self.env.state_dim] = [index+self.collection, index+self.collection]
						# self.buffer[index, self.env.state_dim: self.env.state_dim + self.env.action_dim] = [index+self.collection]
						# self.buffer[index, -3] = index+self.collection
						# self.buffer[index, -2] = index+self.collection
						'''测试共享内存'''

						self.env.current_state = self.env.next_state.copy()
						action_from_actor, s, a_log_prob, s_value = self.choose_action(self.env.current_state)  # 返回三个没有梯度的 tensor
						action_from_actor = action_from_actor.numpy()
						action = self.action_linear_trans(action_from_actor.flatten())  # flatten 也有去掉多余括号的功能
						self.env.step_update(action)
						sumr += self.env.reward

						self.buffer[index, 0: self.env.state_dim] = self.env.current_state[:]
						self.buffer[index, self.env.state_dim: self.env.state_dim + self.env.action_dim] = action_from_actor[:]
						self.buffer[index, -3] = a_log_prob.item()
						self.buffer[index, -2] = s_value.item()
						_immediate_r[index] = self.env.reward
						_done[index] = 1.0 if self.env.is_terminal else 0.0

						index += 1
						self.timestep += 1
						if self.timestep % self.action_std_decay_freq == 0:
							self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
						if index == self.buffer_size:
							break
					self.episode += 1
				print('Finish collecting data, average reward:', round(sumr / (self.episode + 1 - start_collection), 3))
				start_collection = self.episode
				# print('----------')
				self.package_rewards(_immediate_r, _done)  # 将 reward 放到 buffer 的最后一列
				self.collection += 1
				self.permit[0] = 0  # 数据采集结束，将标志置 0
			else:
				pass
		print('Data collection done...')
		self.permit[0] = 0

class Distributed_PPO2(mp.Process):
	def __init__(self,
				 env,
				 policy: PPOActorCritic,
				 actor_lr: float = 3e-4,
				 critic_lr: float = 1e-3,
				 k_epo: int = 250,
				 num_of_pro: int = 5,
				 eps_clip: float = 0.2,
				 buffer_size: int = 1200,
				 total_tr_cnt: int = 5000,
				 ref_buffer: np.ndarray = np.array([]),
				 share_buffer: list = None,
				 ref_permit: np.ndarray = np.array([]),
				 share_permit: list = None,
				 policy_net_shape: list = None,
				 share_net_list: list = None,
				 path: str = ''):
		super(Distributed_PPO2, self).__init__()
		setup_seed(3407)
		if share_buffer is None:
			share_buffer = []
		if share_permit is None:
			share_permit = []
		if policy_net_shape is None:
			policy_net_shape = []
		if share_net_list is None:
			share_net_list = []
		self.env = env
		self.path = path
		self.device = 'cpu'	# cuda:0
		self.policy = policy
		self.policy.to(self.device)
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.optimizer = torch.optim.Adam([
			{'params': self.policy.actor.parameters(), 'lr': self.actor_lr},
			{'params': self.policy.critic.parameters(), 'lr': self.critic_lr}
		])
		self.loss = nn.MSELoss()
		self.k_epo = k_epo
		self.eps_clip = eps_clip
		self.total_tr_cnt = total_tr_cnt
		self.training = 0  # 记录已经训练的次数
		self.num_of_pro = num_of_pro
		self.buffer_size = buffer_size
		self.buffer = []
		self.permit = []
		self.net_param = []
		self.policy_net_shape = policy_net_shape

		assert (len(share_buffer) == self.num_of_pro)

		'''创建了 policy 探索数据的共享内存'''
		for i in range(self.num_of_pro):
			self.buffer.append(np.ndarray(ref_buffer.shape, ref_buffer.dtype, share_buffer[i]))		# 对应每一个 worker 的共享内存
			self.permit.append(np.ndarray(ref_permit.shape, ref_permit.dtype, share_permit[i]))		# 对应每一个 buffer 的标志位
		'''创建了 policy 探索数据的共享内存'''

		'''创建了网络参数共享的共享内存'''
		for _shape, _buf in zip(self.policy_net_shape, share_net_list):
			self.net_param.append(np.ndarray(_shape, np.float32, _buf))
		self.package_net_params2worker()
		'''创建了网络参数共享的共享内存'''

		'''用于学习的数据'''
		self.data_s = np.zeros((self.buffer_size * self.num_of_pro, self.env.state_dim))
		self.data_a = np.zeros((self.buffer_size * self.num_of_pro, self.env.action_dim))
		self.data_lg_prob = np.zeros(self.buffer_size * self.num_of_pro)
		self.data_vs = np.zeros(self.buffer_size * self.num_of_pro)
		self.data_r = np.zeros(self.buffer_size * self.num_of_pro)
		'''用于学习的数据'''

	def is_train_permit(self):
		for i in range(self.num_of_pro):
			if self.permit[i][0] == 1:
				return False
		return True

	def permit_exploration(self):
		for i in range(self.num_of_pro):
			self.permit[i][0] = 1  # 允许子进程开始搜集数据

	def forbid_exploration(self):
		for i in range(self.num_of_pro):
			self.permit[i][0] = 0  # 禁止子进程开始搜集数据

	def evaluate(self, state):
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
			action_mean = self.policy.actor(t_state)
		return action_mean.detach()

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
		return linear_action

	def print_data_in_worker(self, i):
		print('worker ', i)
		print('state: ', self.buffer[i])
		print('action: ', self.data_a[i * self.buffer_size: (i + 1) * self.buffer_size])
		print('lg_prob: ', self.data_lg_prob[i * self.buffer_size: (i + 1) * self.buffer_size])
		print('vs: ', self.data_vs[i * self.buffer_size: (i + 1) * self.buffer_size])
		print('r: ', self.data_r[i * self.buffer_size: (i + 1) * self.buffer_size])

	def package_net_params2worker(self):
		for i, param in zip(range(len(self.policy_net_shape)), self.policy.parameters()):
			self.net_param[i][:] = param.detach().cpu().numpy()[:]		# 将网络参数送到对应的共享内存中

	def run(self) -> None:
		while self.training < self.total_tr_cnt:	# 当没有完成全部训练
			if self.is_train_permit():	# 如果允许开始训练
				print('starting training')
				self.policy.to(self.device)

				'''2. 把数据拿过来，维度对应好'''
				data = np.vstack(self.buffer)		# 把所有数据垂直合并在一起
				self.data_s = data[:, 0: self.env.state_dim]
				self.data_a = data[:, self.env.state_dim: self.env.state_dim + self.env.action_dim]
				self.data_lg_prob = data[:, -3]
				self.data_vs = data[:, -2]
				self.data_r = data[:, -1]

				'''3. numpy 变 tensor'''
				rewards = torch.FloatTensor(self.data_r).detach().to(self.device)
				rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 奖励归一化

				with torch.no_grad():
					old_states = torch.FloatTensor(self.data_s).detach().to(self.device)
					old_actions = torch.FloatTensor(self.data_a).detach().to(self.device)
					old_log_probs = torch.FloatTensor(self.data_lg_prob).detach().to(self.device)
					old_state_values = torch.FloatTensor(self.data_vs).detach().to(self.device)

				'''4. network training'''
				advantages = rewards.detach() - old_state_values.detach()
				for _ in range(self.k_epo):
					log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
					state_values = torch.squeeze(state_values)
					ratios = torch.exp(log_probs - old_log_probs.detach())
					surr1 = ratios * advantages
					surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
					loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy
					self.optimizer.zero_grad()
					loss.mean().backward()
					self.optimizer.step()
				'''5. package net parameters to shared memory'''
				self.package_net_params2worker()
				'''6. 学习次数加一'''
				print('Training {} finished...'.format(self.training))
				self.training += 1

				'''6. 训练 50 次，就测试一下'''
				if self.training % 50 == 0:
					self.save_models()
					test_num = 5
					print('evaluating...')
					for _ in range(test_num):
						self.env.reset_random()
						while not self.env.is_terminal:
							self.env.current_state = self.env.next_state.copy()
							action_from_actor = self.evaluate(self.env.current_state)
							action_from_actor = action_from_actor.numpy()
							action = self.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
							self.env.step_update(action)  # 环境更新的action需要是物理的action
							# r += self.env.reward
							self.env.show_dynamic_image(isWait=False)  # 画图
					cv.destroyAllWindows()

				'''7. 打开 flag，再次允许 worker 收集数据'''
				self.permit_exploration()
			else:						# 如果不允许，说明 worker 正在采集数据，chief 啥也不用干，等着就行
				pass
		print('Training terminate...')
		time.sleep(1)

	def save_models(self):
		self.policy.save_checkpoint()

	def save_models_all(self):
		self.policy.save_all_net()

	def DPPO2_info(self):
		with open(self.path + 'DPPO2_info.txt', 'w') as f:
			f.writelines('========== DPPO2 info ==========')
			f.writelines('number of process: {}'.format(self.num_of_pro))
			f.writelines('agent name: {}'.format(self.env.name))
			f.writelines('state_dim: {}'.format(self.env.state_dim))
			f.writelines('action_dim: {}'.format(self.env.action_dim))
			f.writelines('action_range: {}'.format(self.env.action_range))
			f.writelines('actor learning rate: {}'.format(self.actor_lr))
			f.writelines('critic learning rate: {}'.format(self.critic_lr))
			f.writelines('DPPO2 training device: {}'.format(self.device))
			f.writelines('total training count: {}'.format(self.total_tr_cnt))
			f.writelines('k_epoch: {}'.format(self.k_epo))
			f.writelines('========== DPPO2 info ==========')
