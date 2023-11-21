import os, sys, datetime
import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from SecondOrderIntegration import SecondOrderIntegration
from Distributed_PPO2 import Distributed_PPO2 as DPPO2
from Distributed_PPO2 import Worker
from utils.classes import Normalization, SharedAdam

ENV = 'SecondOrderIntegration'
ALGORITHM = 'PPO2'


def setup_seed(seed):
	torch.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)

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
		self.fc1 = nn.Linear(state_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, action_dim)
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
		self.orthogonal_init(self.fc3, gain=0.01)

	def forward(self, s):
		s = torch.tanh(self.fc1(s))
		s = torch.tanh(self.fc2(s))
		s = torch.tanh(self.fc3(s))
		s = s * self.gain + self.off
		return s

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
		self.net = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1),
		)
		self.init(use_orthogonal_init=use_orthogonal_init)

	@staticmethod
	def orthogonal_init(layer, gain=1.0):
		nn.init.orthogonal_(layer.weight, gain=gain)
		nn.init.constant_(layer.bias, 0)

	def orthogonal_init_all(self):
		self.orthogonal_init(self.net[0])
		self.orthogonal_init(self.net[2])
		self.orthogonal_init(self.net[4])

	def forward(self):
		raise NotImplementedError

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

	process_num = 20
	actor_lr = 1e-4  / min(process_num, 5)
	critic_lr = 1e-3  / min(process_num, 5)  # 一直都是 1e-3
	# k_epo = int(100 / process_num * 1)  # int(100 / process_num * 1.1)
	k_epo = 30 / min(process_num, 5)
	agent = DPPO2(env=env, actor_lr=actor_lr, critic_lr=critic_lr, num_of_pro=process_num, path=simulationPath)

	'''3. 重新加载全局网络和优化器，这是必须的操作'''
	agent.global_actor = PPOActor_Gaussian(state_dim=env.state_dim,
										   action_dim=env.action_dim,
										   a_min=np.array(env.action_range)[:, 0],
										   a_max=np.array(env.action_range)[:, 1],
										   init_std=1.2,
										   use_orthogonal_init=True).share_memory()
	agent.global_critic = PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True)
	agent.eval_actor = PPOActor_Gaussian(state_dim=env.state_dim,
										 action_dim=env.action_dim,
										 a_min=np.array(env.action_range)[:, 0],
										 a_max=np.array(env.action_range)[:, 1],
										 init_std=1.2,
										 use_orthogonal_init=True).share_memory()

	if RETRAIN:
		agent.global_actor.load_state_dict(torch.load('Policy_PPO_4_20700'))
		agent.global_critic.load_state_dict(torch.load('Policy_PPO_4_20700'))

	# agent.global_actor.share_memory()
	# agent.global_critic.share_memory()

	agent.actor_optimizer = SharedAdam([{'params': agent.global_actor.parameters(), 'lr': actor_lr}], eps=1e-5)
	agent.critic_optimizer = SharedAdam([{'params': agent.global_critic.parameters(), 'lr': critic_lr}], eps=1e-5)

	'''4. 添加进程'''
	ppo_msg = {'gamma': 0.99,
			   'k_epo': k_epo,
			   'eps_clip': 0.2,
			   # 'buffer_size': int(env.time_max / env.dt) * 2,
			   'buffer_size': 2048,
			   'state_dim': env.state_dim,
			   'action_dim': env.action_dim,
			   # 'a_lr': 1e-4,
			   # 'c_lr': 1e-3,
			   'device': 'cpu',
			   'set_adam_eps': True,
			   'lmd': 0.95,
			   'use_adv_norm': True,
			   'mini_batch_size': 64,
			   'entropy_coef': 0.01,
			   'use_grad_clip': False,
			   'use_lr_decay': True,
			   'max_train_steps': int(5e6),
			   'using_mini_batch': False}
	for i in range(agent.num_of_pro):
		w = Worker(g_actor=agent.global_actor,
				   l_actor=PPOActor_Gaussian(state_dim=env.state_dim,
											 action_dim=env.action_dim,
											 a_min=np.array(env.action_range)[:, 0],
											 a_max=np.array(env.action_range)[:, 1],
											 init_std=1.2,
											 use_orthogonal_init=True),
				   g_critic=agent.global_critic,
				   l_critic=PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True),
				   g_opt_critic=agent.critic_optimizer,
				   g_opt_actor=agent.actor_optimizer,
				   g_train_n=agent.global_training_num,
				   _index=i,
				   _name='worker' + str(i),
				   _env=env,  # 或者直接写env，随意
				   _queue=agent.queue,
				   _lock=agent.lock,
				   _ppo_msg=ppo_msg)
		agent.add_worker(w)
	agent.DPPO2_info()

	agent.start_multi_process()
