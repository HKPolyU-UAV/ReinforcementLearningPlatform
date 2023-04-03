import os
import sys
import datetime
import time
import cv2 as cv
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from environment.envs.SecondOrderIntegration.SecondOrderIntegration import SecondOrderIntegration as env
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_cls import *

optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ALGORITHM = 'PPO'
ENV = 'SecondOrderIntegration'


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


# setup_seed(3407)


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		"""
		@param _state_dim:
		@param _action_dim:
		@param _action_std_init:
		@param name:
		@param chkpt_dir:
		"""
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		self.state_dim = _state_dim
		self.action_std_init = _action_std_init
		# 应该是初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
		self.action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init)	# 是一个向量
		self.actor = nn.Sequential(
			nn.Linear(self.state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, self.action_dim),
			nn.Tanh()
		)
		self.critic = nn.Sequential(
			nn.Linear(self.state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 32),
			nn.ReLU(),
			nn.Linear(32, 1)
		)
		# self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = 'cpu'
		# torch.cuda.empty_cache()
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

	env = env(pos0=np.array([1.0, 1.0]),
			  vel0=np.array([0.0, 0.0]),
			  map_size=np.array([5.0, 5.0]),
			  target=np.array([4.0, 4.0]))

	if TRAIN:
		action_std_init = 0.8
		'''重新加载Policy网络结构，这是必须的操作'''
		policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulationPath)
		policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulationPath)
		agent = PPO(env=env,
					actor_lr=3e-4,
					critic_lr=1e-3,
					gamma=0.99,
					K_epochs=200,
					eps_clip=0.2,
					action_std_init=action_std_init,
					buffer_size=int(env.timeMax / env.dt * 2),  # 假设可以包含两条完整的最长时间的轨迹
					policy=policy,
					policy_old=policy_old,
					path=simulationPath)
		'''重新加载Policy网络结构，这是必须的操作'''

		agent.PPO_info()

		max_training_timestep = int(env.timeMax / env.dt) * 10000  # 10000回合
		action_std_decay_freq = int(5e6)
		action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
		min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

		sumr = 0
		start_eps = 0
		train_num = 0
		test_num = 0
		index = 0
		USE_BUFFER1 = False
		while timestep <= max_training_timestep:
			env.reset_random()
			if USE_BUFFER1:
				while not env.is_terminal:
					env.current_state = env.next_state.copy()
					# print(env.current_state)
					action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)  # 返回三个没有梯度的tensor
					action_from_actor = action_from_actor.numpy()
					action = agent.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
					env.step_update(action)  # 环境更新的action需要是物理的action
					# env.show_dynamic_image(isWait=False)  # 画图
					sumr += env.reward
					'''存数'''
					agent.buffer.append(s=env.current_state,
										a=action_from_actor,
										log_prob=a_log_prob.numpy(),
										r=env.reward,
										sv=s_value.numpy(),
										done=1.0 if env.is_terminal else 0.0,
										index=index)
					index += 1
					timestep += 1
					'''存数'''
					'''学习'''
					if timestep % agent.buffer.batch_size == 0:
						print('========== LEARN ==========')
						print('Episode: {}'.format(agent.episode))
						print('Num of learning: {}'.format(train_num))
						agent.learn()
						'''clear buffer'''
						# agent.buffer.clear()
						average_train_r = round(sumr / (agent.episode + 1 - start_eps), 3)
						print('Average reward:', average_train_r)
						# agent.writer.add_scalar('train_r', average_train_r, train_num)		# to tensorboard
						train_num += 1
						start_eps = agent.episode
						sumr = 0
						index = 0
						if train_num % 50 == 0 and train_num > 0:
							average_test_r = agent.agent_evaluate(5)
							# agent.writer.add_scalar('test_r', average_test_r, test_num)	# to tensorboard
							test_num += 1
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
			else:
				_temp_s = np.atleast_2d([]).astype(np.float32)
				_temp_a = np.atleast_2d([]).astype(np.float32)
				_temp_log_prob = np.atleast_1d([]).astype(np.float32)
				_temp_r = np.atleast_1d([]).astype(np.float32)
				_temp_sv = np.atleast_1d([]).astype(np.float32)
				_temp_done = np.atleast_1d([]).astype(np.float32)
				_localTimeStep = 0
				sumr = 0
				while not env.is_terminal:
					env.current_state = env.next_state.copy()
					action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)  # 返回三个没有梯度的tensor
					action_from_actor = action_from_actor.numpy()
					action = agent.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
					env.step_update(action)  # 环境更新的action需要是物理的action
					# env.show_dynamic_image(isWait=False)  # 画图
					sumr += env.reward
					'''临时存数'''
					if len(_temp_done) == 0:
						_temp_s = np.atleast_2d(env.current_state).astype(np.float32)
						_temp_a = np.atleast_2d(action_from_actor).astype(np.float32)
						_temp_log_prob = np.atleast_1d(a_log_prob.numpy()).astype(np.float32)
						_temp_r = np.atleast_1d(env.reward).astype(np.float32)
						_temp_sv = np.atleast_1d(s_value.numpy()).astype(np.float32)
						_temp_done = np.atleast_1d(1.0 if env.is_terminal else 0.0).astype(np.float32)
					else:
						_temp_s = np.vstack((_temp_s, env.current_state))
						_temp_a = np.vstack((_temp_a, action_from_actor))
						_temp_log_prob = np.hstack((_temp_log_prob, a_log_prob.numpy()))
						_temp_r = np.hstack((_temp_r, env.reward))
						_temp_sv = np.hstack((_temp_sv, s_value.numpy()))
						_temp_done = np.hstack((_temp_done, 1.0 if env.is_terminal else 0.0))
					_localTimeStep += 1
					'''临时存数'''

				agent.episode += 1

				'''学习'''
				if env.terminal_flag == 2:		# 首先，必须是不出界的轨迹
				# if True:
					print('得到一条轨迹，添加...')
					print('cumulative reward: ', sumr)
					timestep += _localTimeStep
					agent.buffer2.append_traj(_temp_s, _temp_a, _temp_log_prob, _temp_r, _temp_sv, _temp_done)
					if agent.buffer2.buffer_size > agent.buffer.batch_size:
						print('========== LEARN ==========')
						print('Num of learning: {}'.format(train_num))
						agent.learn()
						agent.buffer2.clean()
						train_num += 1
						if train_num % 50 == 0 and train_num > 0:
							average_test_r = agent.agent_evaluate(5)
							test_num += 1
							print('check point save')
							temp = simulationPath + 'train_num' + '_' + str(train_num) + '_save/'
							os.mkdir(temp)
							time.sleep(0.001)
							agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
						print('========== LEARN ==========')
				'''学习'''

				if train_num % 500 == 0 and train_num > 0:		# 训练 200 次，减小方差
					print('减小方差')
					agent.decay_action_std(action_std_decay_rate, min_action_std)

	if TEST:
		pass
