import matplotlib.pyplot as plt
from FNTSMC import fntsmc_param
from ref_cmd import *
from uav import uav_param
from uav_att_ctrl_RL import uav_att_ctrl_RL
from utils.functions import *

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
uav_param.pos_zone = np.atleast_2d([[-5, 5], [-5, 5], [0, 3]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-45), deg2rad(45)], [deg2rad(-45), deg2rad(45)], [deg2rad(-120), deg2rad(120)]])
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
att_ctrl_param.saturation = np.array([0.3, 0.3, 0.3])  # max torque
'''Parameter list of the attitude controller'''

if __name__ == '__main__':
	'''1. Define a controller'''
	att_ctrl_rl = uav_att_ctrl_RL(uav_param, att_ctrl_param)

	NUM_OF_SIMULATION = 5
	cnt = 0

	while cnt < NUM_OF_SIMULATION:
		att_ctrl_rl.reset_uav_att_ctrl(random_att_trajectory=False, yaw_fixed=False, new_att_ctrl_param=None)
		if cnt % 1 == 0:
			print('Current:', cnt)

		'''3. Control'''
		while att_ctrl_rl.time < att_ctrl_rl.time_max - DT / 2:
			'''3.1. generate reference signal'''
			rhod, dot_rhod, dot2_rhod, _ = ref_inner(att_ctrl_rl.time,
													 att_ctrl_rl.ref_att_amplitude,
													 att_ctrl_rl.ref_att_period,
													 att_ctrl_rl.ref_att_bias_a,
													 att_ctrl_rl.ref_att_bias_phase)

			'''3.2. control'''
			# torque = att_ctrl.att_control(ref=rhod, dot_ref=dot_rhod, dot2_ref=dot2_rhod)
			torque = att_ctrl_rl.att_control(ref=rhod, dot_ref=dot_rhod, dot2_ref=None)
			att_ctrl_rl.update(action=torque)

			att_ctrl_rl.visualization()

		cnt += 1
		# print('Finish...')
		# SAVE = False
		# if SAVE:
		# 	new_path = (os.path.dirname(os.path.abspath(__file__)) +
		# 				'/../../datasave/' +
		# 				datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/')
		# 	os.mkdir(new_path)
		# 	att_ctrl.collector.package2file(path=new_path)

		att_ctrl_rl.collector.plot_att()
		att_ctrl_rl.collector.plot_dot_att()
		att_ctrl_rl.collector.plot_pqr()
		att_ctrl_rl.collector.plot_torque()
		plt.show()
