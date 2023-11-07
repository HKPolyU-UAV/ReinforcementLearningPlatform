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
from utils.functions import *
from utils.classes import PPOActor_Gaussian


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
	env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
	env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)
	opt_actor = PPOActor_Gaussian(state_dim=env.state_dim,
								  action_dim=env.action_dim,
								  a_min=np.array(env.action_range)[:, 0],
								  a_max=np.array(env.action_range)[:, 1],
								  init_std=0.5,
								  use_orthogonal_init=True)
	optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/net/'
	opt_actor.load_state_dict(torch.load(optPath + 'actor'))  # 测试时，填入测试actor网络
	env.load_norm_normalizer_from_file(optPath, 'state_norm.csv')
	# exit(0)
	n = 1
	for i in range(n):
		opt_SMC_para = np.atleast_2d(np.zeros(env.action_dim))
		reset_pos_ctrl_param('zero')
		env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
												random_pos0=False,
												new_att_ctrl_param=None,
												new_pos_ctrl_parma=pos_ctrl_param,
												outer_param=None)
		env.show_image(False)
		test_r = 0.
		while not env.is_terminal:
			new_SMC_param = opt_actor.evaluate(env.current_state_norm(env.current_state, update=False))
			opt_SMC_para = np.insert(opt_SMC_para, opt_SMC_para.shape[0], new_SMC_param, axis=0)
			env.get_param_from_actor(new_SMC_param)  # 将控制器参数更新
			a_4_uav = env.generate_action_4_uav()
			env.step_update(a_4_uav)
			test_r += env.reward

			env.visualization()
		print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
		# print(opt_SMC_para.shape)
		(pd.DataFrame(opt_SMC_para,
					  columns=['k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'gamma', 'lambda']).
		 to_csv(simulationPath + 'opt_smc_param.csv', sep=',', index=False))

		env.collector.package2file(simulationPath)
		# env.collector.plot_att()
		#
		# env.collector.plot_pos()
		# env.collector.plot_vel()

		# opt_SMC_para = np.delete(opt_SMC_para, 0, axis=0)
		# xx = np.arange(opt_SMC_para.shape[0])
		# plt.figure()
		# plt.grid(True)
		# plt.plot(xx, opt_SMC_para[:, 0], label='k11')
		# plt.plot(xx, opt_SMC_para[:, 1], label='k12')
		# plt.plot(xx, opt_SMC_para[:, 2], label='k13')
		# plt.plot(xx, opt_SMC_para[:, 3], label='k21')
		# plt.plot(xx, opt_SMC_para[:, 4], label='k22')
		# plt.plot(xx, opt_SMC_para[:, 5], label='k23')
		# plt.plot(xx, opt_SMC_para[:, 6], label='gamma')
		# plt.plot(xx, opt_SMC_para[:, 7], label='lambda')
		# plt.legend()
		#
		# plt.show()
