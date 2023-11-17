import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from uav_att_ctrl_RL import uav_att_ctrl_RL
from environment.UavFntsmcParam.uav import uav_param
from environment.UavFntsmcParam.FNTSMC import fntsmc_param
from environment.UavFntsmcParam.ref_cmd import *
from utils.functions import *
from utils.classes import PPOActor_Gaussian


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
uav_param.pos_zone = np.atleast_2d([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-90), deg2rad(90)], [deg2rad(-90), deg2rad(90)], [deg2rad(-180), deg2rad(180)]])
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

test_episode = []
test_reward = []
sumr_list = []


def reset_att_ctrl_param(flag: str):
	if flag == 'zero':
		att_ctrl_param.k1 = 0.01 * np.ones(3)
		att_ctrl_param.k2 = 0.01 * np.ones(3)
		att_ctrl_param.gamma = 0.01 * np.ones(3)
		att_ctrl_param.lmd = 0.01 * np.ones(3)
	elif flag == 'random':
		att_ctrl_param.k1 = np.random.random(3)
		att_ctrl_param.k2 = np.random.random(3)
		att_ctrl_param.gamma = np.random.random() * np.ones(3)
		att_ctrl_param.lmd = np.random.random() * np.ones(3)
	else:  # optimal
		att_ctrl_param.k1 = np.array([25, 25, 40])
		att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
		att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
		att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])


if __name__ == '__main__':
	env = uav_att_ctrl_RL(uav_param, att_ctrl_param)
	env.reset_uav_att_ctrl_RL_tracking(random_trajectory=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)

	opt_actor = PPOActor_Gaussian(state_dim=env.state_dim,
								  action_dim=env.action_dim,
								  a_min=np.array(env.action_range)[:, 0],
								  a_max=np.array(env.action_range)[:, 1],
								  init_std=0.01,
								  use_orthogonal_init=True)
	optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/net/'
	opt_actor.load_state_dict(torch.load(optPath + 'actor'))  # 测试时，填入测试actor网络
	env.load_norm_normalizer_from_file(optPath, 'state_norm.csv')

	n = 10
	for i in range(10):
		reset_att_ctrl_param('zero')
		yyf = [deg2rad(80) * np.ones(3), 5 * np.ones(3), np.array([0, -np.pi / 2, np.pi / 2])]
		env.reset_uav_att_ctrl_RL_tracking(random_trajectory=True,
										   yaw_fixed=False,
										   new_att_ctrl_param=att_ctrl_param,
										   outer_param=None)
		test_r = 0.
		while not env.is_terminal:
			new_SMC_param = opt_actor.evaluate(env.current_state_norm(env.current_state, update=False))
			env.get_param_from_actor(new_SMC_param)  # 将控制器参数更新
			rhod, dot_rhod, _, _ = ref_inner(env.time,
											   env.ref_att_amplitude,
											   env.ref_att_period,
											   env.ref_att_bias_a,
											   env.ref_att_bias_phase)
			torque = env.att_control(rhod, dot_rhod, None)
			env.step_update([torque[0], torque[1], torque[2]])
			test_r += env.reward

			env.visualization()
		print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
