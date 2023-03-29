import os
import sys
import datetime
import time
import cv2 as cv
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from environment.envs.FlightAttitudeSimulator.flight_attitude_simulator_2state_continuous import Flight_Attitude_Simulator_2State_Continuous as FAS_2S
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_func import *
from common.common_cls import *

optPath = '../../../datasave/network/'
timestep = 0
ALGORITHM = 'PPO'
ENV = 'FlightAttitudeSimulator2StateContinuous'


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		# 应该是初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
		self.action_var = torch.full((_action_dim,), _action_std_init * _action_std_init)
		self.actor = nn.Sequential(
			nn.Linear(_state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, _action_dim),
			nn.Tanh()
		)
		self.critic = nn.Sequential(
			nn.Linear(_state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)
		# self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = 'cpu'
		# torch.cuda.empty_cache()
		self.to(self.device)

	def set_action_std(self, new_action_std):
		self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

	def forward(self):
		raise NotImplementedError

	def act(self, _s):
		action_mean = self.actor(_s)
		cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
		dist = MultivariateNormal(action_mean, cov_mat)

		_a = dist.sample()
		action_logprob = dist.log_prob(_a)
		state_val = self.critic(_s)

		return _a.detach(), action_logprob.detach(), state_val.detach()

	def evaluate(self, _s, a):
		action_mean = self.actor(_s)
		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(self.device)
		dist = MultivariateNormal(action_mean, cov_mat)

		# For Single Action Environments.
		if self.action_dim == 1:
			a = a.reshape(-1, self.action_dim)

		action_logprobs = dist.log_prob(a)
		dist_entropy = dist.entropy()
		state_values = self.critic(_s)

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


if __name__ == '__main__':
	log_dir = '../../../datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)
	TRAIN = True  # 直接训练
	RETRAIN = False  # 基于之前的训练结果重新训练
	TEST = not TRAIN

	env = FAS_2S(deg2rad(0), False)

	if TRAIN:
		action_std_init = 0.6
		policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulationPath)
		policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulationPath)
		agent = PPO(env=env,
					actor_lr=3e-4,
					critic_lr=1e-3,
					gamma=0.99,
					K_epochs=80,
					eps_clip=0.2,
					action_std_init=action_std_init,
					buffer_size=int(env.timeMax / env.dt * 2),
					policy=policy,
					policy_old=policy_old,
					path=simulationPath)

		agent.PPO_info()

		learn_every_n_timestep = int(env.timeMax / env.dt) * 2		# 每采集这么多的数据统一学习一次，长度为每回合最大数据的 5 倍
		max_training_timestep = int(env.timeMax / env.dt) * 10000	# 10000回合
		action_std_decay_freq = int(2.5e5)
		action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
		min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

		sumr = 0
		start_eps = 0
		train_num = 0
		index = 0
		while timestep <= max_training_timestep:
			env.reset_random()
			while not env.is_terminal:
				env.current_state = env.next_state.copy()
				action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)
				action = agent.action_linear_trans(action_from_actor.detach().cpu().numpy().flatten())
				env.step_update(action)
				sumr += env.reward
				'''存数'''
				agent.buffer.append(s=env.current_state,
									a=action_from_actor,
									log_prob=a_log_prob.numpy(),
									r=env.reward,
									sv=s_value.numpy(),
									done=1.0 if env.is_terminal else 0.0,
									index=index)
				'''存数'''
				index += 1
				timestep += 1
				'''学习'''
				if timestep % learn_every_n_timestep == 0:
					print('========== LEARN ==========')
					print('Episode: {}'.format(agent.episode))
					print('Num of learning: {}'.format(train_num))
					agent.learn()
					index = 0
					train_num += 1
					print('Average reward:', round(sumr / (agent.episode + 1 - start_eps), 3))
					start_eps = agent.episode
					sumr = 0
					if train_num % 50 == 0 and train_num > 0:		# 每学习 50 次，测试一下下，看看效果~ ^_^
						agent.agent_evaluate(5)
						print('check point save')
						temp = simulationPath + 'episode' + '_' + str(agent.episode) + '_save/'
						os.mkdir(temp)
						time.sleep(0.01)
						agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
					print('========== LEARN ==========')
				'''学习'''

				if timestep % action_std_decay_freq == 0:
					agent.decay_action_std(action_std_decay_rate, min_action_std)
			agent.episode += 1

	if TEST:
		pass
