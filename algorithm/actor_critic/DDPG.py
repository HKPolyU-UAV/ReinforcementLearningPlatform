from environment.config.xml_write import xml_cfg
from common.common_func import *
from common.common_cls import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class DDPG:
	def __init__(self,
				 env,
				 gamma: float = 0.9,
				 actor_soft_update: float = 1e-2,
				 critic_soft_update: float = 1e-2,
				 memory_capacity: int = 5000,
				 batch_size: int = 64,
				 actor: Actor = Actor(),
				 target_actor: Actor = Actor(),
				 critic: Critic = Critic(),
				 target_critic: Critic = Critic(),
				 path: str = ''):
		"""
        @param env:
        @param gamma:
        @param actor_soft_update:
        @param critic_soft_update:
        @param memory_capacity:
        @param batch_size:
        @param actor:
        @param target_actor:
        @param critic:
        @param target_critic:
        @param path:
        """
		self.env = env

		'''DDPG'''
		self.gamma = gamma
		self.actor_tau = actor_soft_update
		self.critic_tau = critic_soft_update
		self.memory = ReplayBuffer(memory_capacity, batch_size, self.env.state_dim, self.env.action_dim)
		self.path = path
		'''DDPG'''

		'''network'''
		self.actor = actor
		self.target_actor = target_actor
		self.critic = critic
		self.target_critic = target_critic
		'''network'''

		self.noise_OU = OUActionNoise(mu=np.zeros(self.env.action_dim))
		self.noise_gaussian = GaussianNoise(mu=np.zeros(self.env.action_dim))
		self.update_network_parameters()

		self.episode = 0
		self.reward = 0

		self.save_episode = []  # 保存的每一个回合的回合数
		self.save_reward = []  # 保存的每一个回合的奖励
		self.save_time = []
		self.save_average_reward = []  # 保存的每一个回合的平均时间的奖励
		self.save_successful_rate = []
		self.save_step = []  # 保存的每一步的步数
		self.save_stepreward = []  # 保存的每一步的奖励

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

	def learn(self, is_reward_ascent=True):
		if self.memory.mem_counter < self.memory.batch_size:
			return

		'''第一步：取数据'''
		state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
		state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
		action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
		reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
		new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
		done = torch.tensor(done, dtype=torch.float).to(self.critic.device)
		'''第一步：取数据'''

		'''第二步：将网络设置到估计模式'''
		self.target_actor.eval()
		self.target_critic.eval()
		self.critic.eval()
		'''第二步：将网络设置到估计模式'''

		'''第三步骤：得到action和Q-Value'''
		target_actions = self.target_actor.forward(new_state)  # a'
		critic_value_ = self.target_critic.forward(new_state, target_actions)  # Q_Target(s', a')
		critic_value = self.critic.forward(state, action)  # Q(s, a)
		'''第三步骤：得到action和Q-Value'''

		# target = []
		# for j in range(self.memory.batch_size):
		#     target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
		# target = torch.tensor(target).to(self.critic.device)
		# target = target.view(self.memory.batch_size, 1)

		target = reward + self.gamma * critic_value_.squeeze() * done
		target = target.to(self.critic.device)
		target = target.view(self.memory.batch_size, 1)

		self.critic.train()
		self.critic.optimizer.zero_grad()
		critic_loss = func.mse_loss(target, critic_value)
		critic_loss.backward()
		self.critic.optimizer.step()

		self.critic.eval()
		self.actor.optimizer.zero_grad()
		mu = self.actor.forward(state)
		self.actor.train()
		actor_loss = -self.critic.forward(state, mu)
		actor_loss = torch.mean(actor_loss)
		actor_loss.backward()
		self.actor.optimizer.step()

		self.update_network_parameters()

	def update_network_parameters(self):
		"""
        :return:        None
        """
		for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.critic_tau) + param.data * self.critic_tau)  # soft update
		for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.actor_tau) + param.data * self.actor_tau)  # soft update

	def save_models(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.target_critic.save_checkpoint()

	def save_models_all(self):
		self.actor.save_all_net()
		self.critic.save_all_net()
		self.target_actor.save_all_net()
		self.target_critic.save_all_net()

	def load_models(self, path):
		"""
        :brief:         only for test
        :param path:    file path
        :return:
        """
		print('...loading checkpoint...')
		self.actor.load_state_dict(torch.load(path + 'Actor_ddpg'))
		self.target_actor.load_state_dict(torch.load(path + 'TargetActor_ddpg'))
		self.critic.load_state_dict(torch.load(path + 'Critic_ddpg'))
		self.target_critic.load_state_dict(torch.load(path + 'TargetCritic_ddpg'))

	def load_actor_optimal(self, path, file):
		print('...loading optimal...')
		self.actor.load_state_dict(torch.load(path + file))

	def load_target_actor_optimal(self, path, file):
		print('...loading optimal...')
		self.target_actor.load_state_dict(torch.load(path + file))

	def get_RLBase_from_XML(self, filename):
		rl_base, agentName = self.load_rl_basefromXML(filename=filename)
		state_dim_nn = int(rl_base['state_dim'])  # input dimension of NN
		action_dim_nn = int(rl_base['action_dim'])
		action_range = str2list(rl_base['action_range'])
		return agentName, state_dim_nn, action_dim_nn, action_range

	@staticmethod
	def load_rl_basefromXML(filename: str) -> (dict, str):
		"""
        :brief:             从模型文件中加载数据到DQN中
        :param filename:    模型文件
        :return:            数据字典
        """
		root = xml_cfg().XML_Load(filename)
		return xml_cfg().XML_GetTagValue(node=xml_cfg().XML_FindNode(nodename='RL_Base', root=root)), root.attrib['name']

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
		return linear_action

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

	def saveData_EpisodeReward(self, episode, time, reward, average_reward, successrate, is2file=False, filename='EpisodeReward.csv'):
		if is2file:
			data = pd.DataFrame({
				'episode': self.save_episode,
				'time': self.save_time,
				'reward': self.save_reward,
				'average_reward': self.save_average_reward,
				'success_rate': self.save_successful_rate
			})
			data.to_csv(self.path + filename, index=False, sep=',')
		else:
			self.save_episode.append(episode)
			self.save_time.append(time)
			self.save_reward.append(reward)
			self.save_average_reward.append(average_reward)
			self.save_successful_rate.append(successrate)
