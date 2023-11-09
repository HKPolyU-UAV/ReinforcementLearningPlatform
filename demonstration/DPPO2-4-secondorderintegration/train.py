import os
import sys
import datetime
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from SecondOrderIntegration import SecondOrderIntegration
from algorithm.policy_base.Distributed_PPO2 import Distributed_PPO2 as DPPO2
from algorithm.policy_base.Distributed_PPO2 import Worker
from utils.classes import Normalization, SharedAdam
import torch.multiprocessing as mp

ENV = 'SecondOrderIntegration'
ALGORITHM = 'PPO2'
test_episode = []
test_reward = []
sumr_list = []

# def setup_seed(seed):
# 	torch.manual_seed(seed)
# 	# torch.cuda.manual_seed_all(seed)
# 	np.random.seed(seed)
# 	random.seed(seed)


# setup_seed(3407)
os.environ["OMP_NUM_THREADS"] = "1"


class PPOActor_Gaussian(nn.Module):
	def __init__(self,
				 state_dim: int = 3,
				 action_dim: int = 3,
				 a_min: np.ndarray = np.zeros(3),
				 a_max: np.ndarray = np.ones(3),
				 init_std: float = 0.5,
				 use_orthogonal_init: bool = True):
		super(PPOActor_Gaussian, self).__init__()
		# self.fc1 = nn.Linear(state_dim, 128)
		# self.fc2 = nn.Linear(128, 128)
		# self.fc3 = nn.Linear(128, 64)
		# self.mean_layer = nn.Linear(64, action_dim)
		self.fc1 = nn.Linear(state_dim, 64)
		self.fc2 = nn.Linear(64, 32)
		self.mean_layer = nn.Linear(32, action_dim)
		self.activate_func = nn.Tanh()
		self.a_min = torch.tensor(a_min, dtype=torch.float)
		self.a_max = torch.tensor(a_max, dtype=torch.float)
		self.off = (self.a_min + self.a_max) / 2.0
		self.gain = self.a_max - self.off
		self.action_dim = action_dim
		self.std = torch.tensor(init_std, dtype=torch.float)

		if use_orthogonal_init:
			self.orthogonal_init_all()

	@staticmethod
	def orthogonal_init(layer, gain=1.0):
		nn.init.orthogonal_(layer.weight, gain=gain)
		nn.init.constant_(layer.bias, 0)

	def orthogonal_init_all(self):
		self.orthogonal_init(self.fc1)
		self.orthogonal_init(self.fc2)
		# self.orthogonal_init(self.fc3)
		self.orthogonal_init(self.mean_layer, gain=0.01)

	def forward(self, s):
		s = self.activate_func(self.fc1(s))
		s = self.activate_func(self.fc2(s))
		# s = self.activate_func(self.fc3(s))
		mean = torch.tanh(self.mean_layer(s)) * self.gain + self.off
		# mean = torch.relu(self.mean_layer(s))
		return mean

	def get_dist(self, s):
		mean = self.forward(s)
		std = self.std.expand_as(mean)
		dist = Normal(mean, std)
		return dist

	def evaluate(self, state):
		with torch.no_grad():
			t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
			action_mean = self.forward(t_state)
		return action_mean.detach().cpu().numpy().flatten()


class PPOCritic(nn.Module):
	def __init__(self, state_dim=3, use_orthogonal_init: bool = True):
		super(PPOCritic, self).__init__()
		# self.fc1 = nn.Linear(state_dim, 128)
		# self.fc2 = nn.Linear(128, 128)
		# self.fc3 = nn.Linear(128, 32)
		# self.fc4 = nn.Linear(32, 1)
		self.fc1 = nn.Linear(state_dim, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 1)
		self.activate_func = nn.Tanh()

		if use_orthogonal_init:
			self.orthogonal_init_all()

	@staticmethod
	def orthogonal_init(layer, gain=1.0):
		nn.init.orthogonal_(layer.weight, gain=gain)
		nn.init.constant_(layer.bias, 0)

	def orthogonal_init_all(self):
		self.orthogonal_init(self.fc1)
		self.orthogonal_init(self.fc2)
		self.orthogonal_init(self.fc3)

	# self.orthogonal_init(self.fc4)

	def forward(self, s):
		s = self.activate_func(self.fc1(s))
		s = self.activate_func(self.fc2(s))
		# s = self.activate_func(self.fc3(s))
		v_s = self.fc3(s)
		return v_s

	def init(self, use_orthogonal_init):
		if use_orthogonal_init:
			self.orthogonal_init_all()
		else:
			self.fc1.reset_parameters()
			self.fc2.reset_parameters()
			self.fc3.reset_parameters()


if __name__ == '__main__':
	log_dir = os.path.dirname(os.path.abspath(__file__)) + '/datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)

	RETRAIN = False

	env = SecondOrderIntegration()
	reward_norm = Normalization(shape=1)

	env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'name': env.name, 'action_range': env.action_range}

	mp.set_start_method('spawn', force=True)

	process_num = 50
	actor_lr = 1e-4  # / min(process_num, 10)
	critic_lr = 1e-3  # / min(process_num, 10)  # 一直都是 1e-3
	action_std = 0.6
	# k_epo = int(100 / process_num * 1)  # int(100 / process_num * 1.1)
	k_epo = 50
	agent = DPPO2(env=env, actor_lr=actor_lr, critic_lr=critic_lr, num_of_pro=process_num, path=simulationPath)

	'''3. 重新加载全局网络和优化器，这是必须的操作，因为考虑到不同的学习环境要设计不同的网络结构，在训练前，要重写 PPOActorCritic 类'''
	agent.global_actor = PPOActor_Gaussian(state_dim=env.state_dim,
										   action_dim=env.action_dim,
										   a_min=np.array(env.action_range)[:, 0],
										   a_max=np.array(env.action_range)[:, 1],
										   init_std=0.8,
										   use_orthogonal_init=True)
	agent.global_critic = PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True)

	if RETRAIN:
		agent.global_actor.load_state_dict(torch.load('Policy_PPO_4_20700'))
		agent.global_critic.load_state_dict(torch.load('Policy_PPO_4_20700'))
	agent.global_actor.share_memory()
	agent.global_critic.share_memory()
	agent.actor_optimizer = SharedAdam([{'params': agent.global_actor.parameters(), 'lr': actor_lr}, ])
	agent.critic_optimizer = SharedAdam([{'params': agent.global_critic.parameters(), 'lr': critic_lr}])

	'''4. 添加进程'''
	ppo_msg = {'gamma': 0.99,
			   'k_epo': 200,
			   'eps_clip': 0.2,
			   'buffer_size': int(env.timeMax / env.dt) * 2,
			   'state_dim': env.state_dim,
			   'action_dim': env.action_dim,
			   'a_lr': 1e-4,
			   'c_lr': 1e-3,
			   'device': 'cpu',
			   'set_adam_eps': True,
			   'lmd': 0.95,
			   'use_adv_norm': True,
			   'mini_batch_size': 64,
			   'entropy_coef': 0.01,
			   'use_grad_clip': True,
			   'use_lr_decay': True,
			   'max_train_steps': int(5e6),
			   'using_mini_batch': False}
	for i in range(agent.num_of_pro):
		w = Worker(g_actor=agent.global_actor,
				   l_actor=PPOActor_Gaussian(state_dim=env.state_dim,
											 action_dim=env.action_dim,
											 a_min=np.array(env.action_range)[:, 0],
											 a_max=np.array(env.action_range)[:, 1],
											 init_std=0.8,
											 use_orthogonal_init=True),
				   g_critic=agent.global_critic,
				   l_critic=PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True),
				   g_opt_critic=agent.critic_optimizer,
				   g_opt_actor=agent.critic_optimizer,
				   g_train_n=agent.global_training_num,
				   _index=i,
				   _name='worker' + str(i),
				   _env=env,  # 或者直接写env，随意
				   _queue=agent.queue,
				   _lock=agent.lock,
				   _ppo_msg=ppo_msg)
		agent.add_worker(w)
	agent.DPPO_info()

	agent.start_multi_process()
