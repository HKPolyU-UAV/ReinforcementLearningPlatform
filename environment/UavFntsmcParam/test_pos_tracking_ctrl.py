import datetime
import os
import sys

from FNTSMC import fntsmc_param
from uav import uav_param
from uav_pos_ctrl_RL import uav_pos_ctrl_RL
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

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
uav_param.pos_zone = np.atleast_2d([[-3, 3], [-3, 3], [0, 3]])
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


if __name__ == '__main__':
    env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)

    NUM_OF_SIMULATION = 1
    cnt = 0
    video = cv.VideoWriter(env.name + '.mp4', cv.VideoWriter_fourcc(*"mp4v"), 60, (env.width, env.height))
    while cnt < NUM_OF_SIMULATION:
        '''生成新的参考轨迹的信息'''
        env.reset_uav_pos_ctrl(random_trajectory=True, random_pos0=True, yaw_fixed=False, new_att_ctrl_param=None, new_pos_ctrl_parma=None)

        if cnt % 1 == 0:
            print('Current:', cnt)

        while env.time < env.time_max - DT / 2:
            action_4_uav = env.generate_action_4_uav()
            env.update(action=np.array(action_4_uav))
            env.visualization()
            video.write(env.image)

        cnt += 1
        SAVE = False
        if SAVE:
            new_path = (os.path.dirname(os.path.abspath(__file__)) +
                        '/../../datasave/' +
                        datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/')
            os.mkdir(new_path)
            env.collector.package2file(path=new_path)
        # env.collector.plot_att()
        # env.collector.plot_pqr()
        # env.collector.plot_dot_att()
        # # env.collector.plot_torque()
        env.collector.plot_pos()
        env.collector.plot_vel()
        # # env.collector.plot_throttle()
        # # env.collector.plot_outer_obs()
        plt.show()
    video.release()
