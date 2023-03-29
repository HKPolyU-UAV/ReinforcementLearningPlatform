import atexit
# import numpy as np
import os
import sys
import datetime
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from environment.envs.cartpole.cartpole_angleonly import CartPoleAngleOnly
from algorithm.policy_base.Distributed_PPO2 import Distributed_PPO2 as DPPO2
from algorithm.policy_base.Distributed_PPO2 import Worker
from common.common_cls import *
import torch.multiprocessing as mp
from multiprocessing import shared_memory

optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ENV = 'DPPO2-CartPoleAngleOnly'


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


setup_seed(3407)
os.environ["OMP_NUM_THREADS"] = "1"


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		self.state_dim = _state_dim
		self.action_std_init = _action_std_init
		self.action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init)
		self.actor = nn.Sequential(
			nn.Linear(self.state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, self.action_dim),
			nn.Tanh()
		)
		self.critic = nn.Sequential(
			nn.Linear(self.state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)
		self.device = 'cpu'
		# torch.cuda.empty_cache()	# 清一下显存，不清暂时也没啥问题，不过应该没坏处
		self.to(self.device)

	def set_action_std(self, new_action_std):
		self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

	def forward(self):
		raise NotImplementedError

	def act(self, s):
		action_mean = self.actor(s)
		cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
		dist = MultivariateNormal(action_mean, cov_mat)

		_a = dist.sample()
		action_logprob = dist.log_prob(_a)
		state_val = self.critic(s)

		return _a.detach(), action_logprob.detach(), state_val.detach()

	def evaluate(self, s, a):
		action_mean = self.actor(s)
		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(self.device)
		dist = MultivariateNormal(action_mean, cov_mat)

		# For Single Action Environments.
		if self.action_dim == 1:
			a = a.reshape(-1, self.action_dim)

		action_logprobs = dist.log_prob(a)
		dist_entropy = dist.entropy()
		state_values = self.critic(s)

		return action_logprobs, state_values, dist_entropy

	def save_checkpoint(self, name=None, path='', num=None):
		print('...saving checkpoint...')
		if name is None:
			torch.save(self.state_dict(), self.checkpoint_file)
		else:
			if num is None:
				torch.save(self.state_dict(), path + name)
			else:
				torch.save(self.state_dict(), path + name + str(num))

	def save_all_net(self):
		print('...saving all net...')
		torch.save(self, self.checkpoint_file_whole_net)

	def load_checkpoint(self):
		print('...loading checkpoint...')
		self.load_state_dict(torch.load(self.checkpoint_file))

	def get_net_shape(self):
		n = []
		for p in self.parameters():
			n.append(p.detach().cpu().numpy().shape)
		return n


if __name__ == '__main__':
	log_dir = '../../../datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)
	TRAIN = True  # 直接训练
	RETRAIN = False  # 基于之前的训练结果重新训练
	TEST = not TRAIN

	env = CartPoleAngleOnly(0, False)

	if TRAIN:
		'''1. 启动多进程，如果用 python 自带的 share_memory 的画，千万不能加，否则 share_memory 不好使'''
		# mp.set_start_method(mp.get_start_method(), force=True)	# 这里是 fork，不是 spawn，所以网络只能使用 cpu

		'''2. 定义全局基本参数'''
		'''2.1. 进程数，buffer 大小'''
		num_of_pro = 1		# 进程数
		buffer_size = int(env.timeMax / env.dt * 2)		# buffer 行数
		# buffer_size = 10		# TODO

		'''2.2. buffer 的共享内存'''
		ref_buffer = np.ones((buffer_size, env.state_dim + env.action_dim + 3), dtype=np.float32)	 # 参考 buffer
		share_buffer = [shared_memory.SharedMemory(create=True, size=ref_buffer.nbytes) for _ in range(num_of_pro)]		# 共享数据

		'''2.3. 工作模式切换共享内存'''
		ref_permit = np.zeros(1, dtype=int)
		share_permit = [shared_memory.SharedMemory(create=True, size=ref_permit.nbytes) for _ in range(num_of_pro)]		# 共享标志位

		'''2.4. 定义 PPO 网络，PPO 网络参数结构， 以及定义 PPO 网络参数共享内存'''
		policy = PPOActorCritic(env.state_dim, env.action_dim, 0.1, 'PPO_Policy', simulationPath)	# 不用写 device，这个在 DPPO2 初始化中会写好
		policy_net_shape = policy.get_net_shape()	# 得到每一层网络参数的结构，需要传进 DPPO2
		share_net_list = []		# 需要传进 DPPO2
		for _shape in policy_net_shape:
			_numpy_buf = np.zeros(shape=_shape, dtype=np.float32)
			share_net_list.append(shared_memory.SharedMemory(create=True, size=_numpy_buf.nbytes))

		'''3. 定义 DPPO2 chief 的基本参数，由于 DPPO2 继承了 mp.Process，所以它天然地就是一个进程'''
		actor_lr = 3e-4
		critic_lr = 1e-3
		eps_clip = 0.2
		total_tr_cnt = 50000	# 5000
		k_epo = 50
		agent = DPPO2(env=env,
					  policy=policy, actor_lr=actor_lr, critic_lr=critic_lr, k_epo=k_epo, eps_clip=eps_clip, buffer_size=buffer_size,
					  num_of_pro=num_of_pro, total_tr_cnt=total_tr_cnt,
					  ref_buffer=ref_buffer, share_buffer=[_s.buf for _s in share_buffer],
					  ref_permit=ref_permit, share_permit=[_s.buf for _s in share_permit],
					  policy_net_shape=policy_net_shape, share_net_list=[_s.buf for _s in share_net_list],
					  path=simulationPath)

		'''4. 添加 worker，由于 worker 继承了 mp.Process，所以它天然地就是一个进程'''
		action_std_decay_freq = int(2.5e5)
		action_std_decay_rate = 0.05
		min_action_std = 0.1
		action_std_init = 0.8
		gamma = 0.99
		worker = []
		for i in range(num_of_pro):
			w = Worker(env=env,
					   policy=PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'worker_Policy', ''),
					   gamma=gamma,
					   action_std_decay_freq=action_std_decay_freq,
					   action_std_decay_rate=action_std_decay_rate,
					   min_action_std=min_action_std,
					   action_std_init=action_std_init,
					   name='worker' + str(i), index=i, total_collection=total_tr_cnt,
					   ref_buffer=ref_buffer, share_buffer=share_buffer[i].buf,
					   ref_permit=ref_permit, share_permit=share_permit[i].buf,
					   policy_net_shape=policy_net_shape, share_net_list=[_s.buf for _s in share_net_list])
			worker.append(w)

		agent.DPPO2_info()

		'''5. 启动多进程'''
		agent.start()
		[_w.start() for _w in worker]

		agent.join()
		[_w.join() for _w in worker]

		'''6. 结束'''
		@atexit.register
		def clean():
			for _sb, _sp, _sn in zip(share_buffer, share_permit, share_net_list):
				_sb.close()
				_sb.unlink()
				_sp.close()
				_sp.unlink()
				_sn.close()
				_sn.unlink()
			print('shared memory clean, termination...')
	else:
		pass
