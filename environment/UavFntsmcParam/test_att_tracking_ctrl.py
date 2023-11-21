import matplotlib.pyplot as plt
from FNTSMC import fntsmc_param
from ref_cmd import *
from uav import uav_param
from uav_att_ctrl_RL import uav_att_ctrl_RL
from utils.functions import *
import cv2 as cv

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
    env = uav_att_ctrl_RL(uav_param, att_ctrl_param)

    NUM_OF_SIMULATION = 1
    cnt = 0
    video = cv.VideoWriter(env.name + '.mp4', cv.VideoWriter_fourcc(*"mp4v"), 60, (env.att_w, env.att_h))
    while cnt < NUM_OF_SIMULATION:
        env.reset_uav_att_ctrl(random_att_trajectory=False, yaw_fixed=False, new_att_ctrl_param=None)
        if cnt % 1 == 0:
            print('Current:', cnt)

        '''3. Control'''
        while env.time < env.time_max - DT / 2:
            '''3.1. generate reference signal'''
            rhod, dot_rhod, dot2_rhod, _ = ref_inner(env.time,
                                                     env.ref_att_amplitude,
                                                     env.ref_att_period,
                                                     env.ref_att_bias_a,
                                                     env.ref_att_bias_phase)

            '''3.2. control'''
            # torque = env.att_control(ref=rhod, dot_ref=dot_rhod, dot2_ref=dot2_rhod)
            torque = env.att_control(ref=rhod, dot_ref=dot_rhod, dot2_ref=None)
            env.update(action=torque)

            env.visualization()
            video.write(env.att_image)
        cnt += 1

        env.collector.plot_att()
        env.collector.plot_dot_att()
        env.collector.plot_pqr()
        env.collector.plot_torque()
        plt.show()
    video.release()
