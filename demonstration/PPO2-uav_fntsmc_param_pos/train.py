import os
import sys
import datetime
import time
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.uav_fntsmc_param.uav_pos_ctrl_RL import uav_pos_ctrl_RL, uav_param
from environment.uav_fntsmc_param.FNTSMC import fntsmc_param
from environment.color import Color
from algorithm.policy_base.Proximal_Policy_Optimization2 import Proximal_Policy_Optimization2 as PPO2
from utils.functions import *
from utils.classes import PPOActor_Gaussian, PPOCritic, Normalization

timestep = 0
ENV = 'uav_fntsmc_param_att'
ALGORITHM = 'PPO'

'''Parameter list of the quadrotor'''
DT = 0.02
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 10
uav_param.pos_zone = np.atleast_2d([[-3, 3], [-3, 3], [0, 3]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-90), deg2rad(90)], [deg2rad(-90), deg2rad(90)], [deg2rad(-120), deg2rad(120)]])
'''Parameter list of the quadrotor'''

'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param()
att_ctrl_param.k1 = np.array([25, 25, 40])
att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
att_ctrl_param.alpha = np.array([2.5, 2.5, 2.5])
att_ctrl_param.beta = np.array([0.99, 0.99, 0.99])
att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
att_ctrl_param.dim = 3
att_ctrl_param.dt = DT
att_ctrl_param.ctrl0 = np.array([0., 0., 0.])
att_ctrl_param.saturation = np.array([0.3, 0.3, 0.3])
'''Parameter list of the attitude controller'''

'''Parameter list of the position controller'''
pos_ctrl_param = fntsmc_param()
pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller'''

test_episode = []
test_reward = []
sumr_list = []


def reset_pos_ctrl_param(flag: str):
	if flag == 'zero':
		pos_ctrl_param.k1 = 0.01 * np.ones(3)
		pos_ctrl_param.k2 = 0.01 * np.ones(3)
		pos_ctrl_param.gamma = 0.01 * np.ones(3)
		pos_ctrl_param.lmd = 0.01 * np.ones(3)
	elif flag == 'random':
		pos_ctrl_param.k1 = np.random.random(3)
		pos_ctrl_param.k2 = np.random.random(3)
		pos_ctrl_param.gamma = np.random.random() * np.ones(3)
		pos_ctrl_param.lmd = np.random.random() * np.ones(3)
	else:  # optimal
		pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
		pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
		pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
		pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])


if __name__ == '__main__':
	log_dir = os.path.dirname(os.path.abspath(__file__)) + '/datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)

	RETRAIN = False

	env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
	reset_pos_ctrl_param('zero')
	env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)
	env.show_image(True)

	env_test = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
	reset_pos_ctrl_param('zero')
	env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)

	reward_norm = Normalization(shape=1)

	env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'name': env.name, 'action_range': env.action_range}

	t_epoch = 0  # 当前训练次数
	test_num = 0
	ppo_msg = {'gamma': 0.99,
			   'K_epochs': 50,
			   'eps_clip': 0.2,
			   'buffer_size': int(env.time_max / env.dt) * 2,
			   'state_dim': env_test.state_dim,
			   'action_dim': env_test.action_dim,
			   'a_lr': 1e-4,
			   'c_lr': 1e-3,
			   'set_adam_eps': True,
			   'lmd': 0.95,
			   'use_adv_norm': True,
			   'mini_batch_size': 64,
			   'entropy_coef': 0.01,
			   'use_grad_clip': True,
			   'use_lr_decay': True,
			   'max_train_steps': int(5e6),
			   'using_mini_batch': False}

	agent = PPO2(env_msg=env_msg,
				 ppo_msg=ppo_msg,
				 actor=PPOActor_Gaussian(state_dim=env.state_dim,
										 action_dim=env.action_dim,
										 a_min=np.array(env.action_range)[:, 0],
										 a_max=np.array(env.action_range)[:, 1],
										 init_std=0.4,  # 第2次学是 0.3
										 use_orthogonal_init=True),
				 critic=PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True))
	agent.PPO2_info()

	if RETRAIN:
		print('RELOADING......')
		'''如果两次奖励函数不一样，那么必须重新初始化 critic'''
		# optPath = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/train_opt/trainNum_300_episode_2/'
		optPath = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/temp/'
		agent.actor.load_state_dict(torch.load(optPath + 'actor_trainNum_300'))  # 测试时，填入测试actor网络
		# agent.critic.load_state_dict(torch.load(optPath + 'critic_trainNum_300'))
		agent.critic.init(True)
		'''如果两次奖励函数不一样，那么必须重新初始化 critic'''

	while True:
		'''1. 初始化 buffer 索引和累计奖励记录'''
		sumr = 0.
		buffer_index = 0
		agent.buffer2.clean()
		'''1. 初始化 buffer 索引和累计奖励记录'''

		'''2. 重新开始一次收集数据'''
		while buffer_index < agent.buffer.batch_size:
			if env.is_terminal:  # 如果某一个回合结束
				reset_pos_ctrl_param('zero')
				# if t_epoch % 10 == 0 and t_epoch > 0:
				print('Sumr:  ', sumr)
				sumr_list.append(sumr)
				sumr = 0.
				yyf_A = 1.5 * np.random.choice([-1, 1], 4)
				yyf_T = 5 * np.ones(4)
				yyf_phi0 = np.pi / 2 * np.random.choice([-1, 0, 1], 4)
				yyf = [yyf_A, yyf_T, yyf_phi0]
				env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
												   random_pos0=False,
												   new_att_ctrl_param=None,
												   new_pos_ctrl_parma=pos_ctrl_param,
												   outer_param=None)
			else:
				env.current_state = env.next_state.copy()  # 此时相当于时间已经来到了下一拍，所以 current 和 next 都得更新
				a, a_log_prob = agent.choose_action(env.current_state)
				# new_SMC_param = agent.action_linear_trans(a)	# a 肯定在 [-1, 1]
				new_SMC_param = a.copy()
				env.get_param_from_actor(new_SMC_param)
				action_4_uav = env.generate_action_4_uav()
				env.step_update(action_4_uav)
				sumr += env.reward
				success = 1.0 if env.terminal_flag == 1 else 0.0
				agent.buffer.append(s=env.current_state_norm(env.current_state, update=True),
									a=a,  # a
									log_prob=a_log_prob,  # a_lp
									# r=env.reward,							# r
									r=reward_norm(env.reward),  # 这里使用了奖励归一化
									s_=env.next_state_norm(env.next_state, update=True),
									done=1.0 if env.is_terminal else 0.0,  # done
									success=success,  # 固定时间内，不出界，就是 success
									index=buffer_index  # index
									)
				buffer_index += 1
		'''2. 重新开始一次收集数据'''

		'''3. 开始一次新的学习'''
		print('~~~~~~~~~~ Training Start~~~~~~~~~~')
		print('Train Epoch: {}'.format(t_epoch))
		# timestep += NUM_OF_TRAJ * env.time_max / env.dt
		timestep += ppo_msg['buffer_size']
		agent.learn(timestep, buf_num=1)  # 使用 RolloutBuffer2
		agent.cnt += 1

		'''4. 每学习 10 次，测试一下'''
		if t_epoch % 10 == 0 and t_epoch > 0:
			n = 5
			print('   Training pause......')
			print('   Testing...')
			for i in range(n):
				reset_pos_ctrl_param('zero')
				_yyf_A = 1.5 * np.random.choice([-1, 1], 4)
				_yyf_T = 5 * np.ones(4)
				_yyf_phi0 = np.pi / 2 * np.random.choice([-1, 0, 1], 4)
				_yyf = [_yyf_A, _yyf_T, _yyf_phi0]
				env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
														random_pos0=False,
														new_att_ctrl_param=None,
														new_pos_ctrl_parma=pos_ctrl_param,
														outer_param=None)
				test_r = 0.
				while not env_test.is_terminal:
					_a = agent.evaluate(env.current_state_norm(env_test.current_state, update=False))
					# _new_SMC_param = agent.action_linear_trans(_a)
					_new_SMC_param = _a.copy()
					env_test.get_param_from_actor(_new_SMC_param)  # 将控制器参数更新
					_a_4_uav = env_test.generate_action_4_uav()
					env_test.step_update(_a_4_uav)
					test_r += env_test.reward
					env_test.image = env_test.image_copy.copy()
					env_test.draw_3d_points_projection(np.atleast_2d([env_test.uav_pos(), env_test.pos_ref]), [Color().Red, Color().DarkGreen])
					env_test.draw_time_error(env_test.uav_pos(), env_test.pos_ref)
					env_test.show_image(False)
				test_num += 1
				test_reward.append(test_r)
				print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
			pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
			pd.DataFrame({'sumr_list': sumr_list}).to_csv(simulationPath + 'sumr_list.csv')
			print('   Testing finished...')
			print('   Go back to training...')
		'''4. 每学习 10 次，测试一下'''

		'''5. 每学习 100 次，减小一次探索概率'''
		if t_epoch % 250 == 0 and t_epoch > 0:
			if agent.actor.std > 0.1:
				agent.actor.std -= 0.05
		'''5. 每学习 100 次，减小一次探索概率'''

		'''6. 每学习 50 次，保存一下 policy'''
		if t_epoch % 50 == 0 and t_epoch > 0:
			# 	average_test_r = agent.agent_evaluate(5)
			test_num += 1
			print('...check point save...')
			temp = simulationPath + 'trainNum_{}/'.format(t_epoch)
			os.mkdir(temp)
			time.sleep(0.01)
			agent.save_ac(msg='', path=temp)
			env.save_state_norm(temp)
		'''6. 每学习 50 次，保存一下 policy'''

		t_epoch += 1
		print('~~~~~~~~~~  Training End ~~~~~~~~~~')
