from environment.UavFntsmcParam.uav import uav_param
from algorithm.rl_base import rl_base
from environment.UavFntsmcParam.uav_pos_ctrl import uav_pos_ctrl, fntsmc_param
import math
import numpy as np
from environment.UavFntsmcParam.ref_cmd import *
from environment.color import Color
from utils.classes import Normalization
import pandas as pd


class uav_pos_ctrl_RL(rl_base, uav_pos_ctrl):
    def __init__(self, _uav_param: uav_param, _uav_att_param: fntsmc_param, _uav_pos_param: fntsmc_param):
        rl_base.__init__(self)
        uav_pos_ctrl.__init__(self, _uav_param, _uav_att_param, _uav_pos_param)

        self.staticGain = 2.0

        '''state limitation'''
        # 并非要求数据一定在这个区间内，只是给一个归一化的系数而已，让 NN 不同维度的数据不要相差太大
        # 不要出现：某些维度的数据在 [-3, 3]，另外维度在 [0.05, 0.9] 这种情况即可
        self.e_pos_max = np.array([3., 3., 3.])
        self.e_vel_max = np.array([3., 3., 3.])
        '''state limitation'''

        '''rl_base'''
        self.use_norm = False   # Regularization is used
        self.name = 'uav_pos_ctrl_RL'
        self.state_dim = 3 + 3  # e_pos e_vel
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]

        self.current_state_norm = Normalization(self.state_dim)
        self.next_state_norm = Normalization(self.state_dim)

        self.current_state = np.zeros(self.state_dim)
        self.next_state = np.zeros(self.state_dim)

        self.action_dim = 3 + 3 + 1 + 1  # 3 for k1, 3 for k2, 1 for gamma, 1 for lambda
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_range = [[0, 5.0] for _ in range(self.action_dim)]
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.isActionContinuous = [True for _ in range(self.action_dim)]
        self.current_action = [0.0 for _ in range(self.action_dim)]

        self.reward = 0.
        self.Q_pos = np.array([1., 1., 1.])  # 位置误差惩罚
        self.Q_vel = np.array([0.05, 0.05, 0.05])  # 速度误差惩罚
        self.R = np.array([0.01, 0.01, 0.01])  # 期望加速度输出 (即控制输出) 惩罚
        self.is_terminal = False
        self.terminal_flag = 0
        '''rl_base'''
        self.draw_init_image()

    def get_state(self) -> np.ndarray:
        e_pos_ = self.uav_pos() - self.pos_ref
        e_vel_ = self.uav_vel() - self.dot_pos_ref
        if self.use_norm:
            e_pos_ = e_pos_ / self.e_pos_max
            e_vel_ = e_vel_ / self.e_vel_max
            state = np.concatenate((e_pos_, e_vel_)) * self.staticGain
        else:
            state = np.concatenate((e_pos_, e_vel_))
        return state

    def draw_init_image(self):
        pass

    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_3d_points_projection(np.atleast_2d([self.uav_pos(), self.pos_ref]), [Color().Red, Color().DarkGreen])
        self.draw_time_error(self.uav_pos(), self.pos_ref)
        self.show_image(False)

    def get_reward(self, param=None):
        """
		@param param:
		@return:
		"""
        # ss = self.inverse_state_norm()
        # _e_pos = ss[0: 3]
        # _e_vel = ss[3: 6]
        _e_pos = self.uav_pos() - self.pos_ref
        _e_vel = self.uav_vel() - self.dot_pos_ref

        '''reward for position error'''
        u_pos = -np.dot(_e_pos ** 2, self.Q_pos)

        '''reward for velocity error'''
        u_vel = -np.dot(_e_vel ** 2, self.Q_vel)

        '''reward for control output'''
        u_acc = -np.dot(self.pos_ctrl.control ** 2, self.R)

        '''reward for att out!!'''
        u_extra = 0.
        if self.terminal_flag == 2:  # 位置出界
            print('Position out')
            '''
				给出界时刻的位置、速度、输出误差的累计
			'''
            _n = (self.time_max - self.time) / self.dt
            u_extra = _n * (u_pos + u_vel + u_acc)

        if self.terminal_flag == 3:
            print('Attitude out')
            '''
				给出界时刻的位置、速度、输出误差的累计
			'''
            _n = (self.time_max - self.time) / self.dt
            u_extra = _n * (u_pos + u_vel + u_acc)

        self.reward = u_pos + u_vel + u_acc + u_extra

    def is_success(self):
        """
		@return:
		"""
        '''
			跟踪控制，暂时不定义 “成功” 的概念，不好说啥叫成功，啥叫失败
			因此置为 False，实际也不调用这个函数即可，学习不成功可考虑再加
		'''
        return False

    def is_Terminal(self, param=None):
        self.terminal_flag = self.get_terminal_flag()
        if self.terminal_flag == 0:  # 普通状态
            self.is_terminal = False
        elif self.terminal_flag == 1:  # 超时
            self.is_terminal = True
        elif self.terminal_flag == 2:  # 位置
            self.is_terminal = True
        elif self.terminal_flag == 3:  # 姿态
            self.is_terminal = True
        else:
            self.is_terminal = False

    def step_update(self, action: list):
        """
		@param action:	这个 action 其实是油门 + 三个力矩
		@return:
		"""
        self.current_action = np.array(action)
        self.current_state = self.get_state()

        self.update(action=self.current_action)
        self.is_Terminal()
        self.next_state = self.get_state()
        self.get_reward()

    def get_param_from_actor(self, action_from_actor: np.ndarray, update_k2: bool = True):
        """
        @param action_from_actor:
        @return:
        """
        if np.min(action_from_actor) < 0:
            print('ERROR!!!!')
        for i in range(3):
            if action_from_actor[i] > 0:
                self.pos_ctrl.k1[i] = action_from_actor[i]
            if action_from_actor[i + 3] > 0 and update_k2:
                self.pos_ctrl.k2[i] = action_from_actor[i + 3]
        if action_from_actor[6] > 0:
            self.pos_ctrl.gamma[:] = action_from_actor[6]  # gamma gamma gamma
        if action_from_actor[7] > 0:
            self.pos_ctrl.lmd[:] = action_from_actor[7]  # lmd lmd lmd

    def set_att_ctrl_from_outer(self, param: np.ndarray):
        self.att_ctrl.k1[:] = param[0: 3]
        self.att_ctrl.k2[:] = param[3: 6]
        self.att_ctrl.gamma[:] = param[6]
        self.att_ctrl.lmd[:] = param[7]

    def reset_uav_pos_ctrl_RL_tracking(self,
                                       random_trajectroy: bool = False,
                                       random_pos0: bool = False,
                                       yaw_fixed: bool = False,
                                       new_att_ctrl_param: fntsmc_param = None,
                                       new_pos_ctrl_parma: fntsmc_param = None,
                                       outer_param: list = None):
        """
        @param outer_param:
		@param yaw_fixed:
		@param random_trajectroy:
		@param random_pos0:
		@param new_att_ctrl_param:
		@param new_pos_ctrl_parma:
		@return:
		"""
        self.reset_uav_pos_ctrl(random_trajectroy, random_pos0, yaw_fixed, new_att_ctrl_param, new_pos_ctrl_parma, outer_param)

        '''RL_BASE'''
        self.current_state = self.get_state()
        self.next_state = self.get_state()
        self.current_action = [0.0 for _ in range(self.action_dim)]
        self.reward = 0.
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

    def save_state_norm(self, path, msg=None):
        data = {
            'cur_n': self.current_state_norm.running_ms.n * np.ones(self.state_dim),
            'cur_mean': self.current_state_norm.running_ms.mean,
            'cur_std': self.current_state_norm.running_ms.std,
            'cur_S': self.current_state_norm.running_ms.S,
            'next_n': self.next_state_norm.running_ms.n * np.ones(self.state_dim),
            'next_mean': self.next_state_norm.running_ms.mean,
            'next_std': self.next_state_norm.running_ms.std,
            'next_S': self.next_state_norm.running_ms.S,
        }
        if msg is None:
            pd.DataFrame(data).to_csv(path + 'state_norm.csv', index=False)
        else:
            pd.DataFrame(data).to_csv(path + 'state_norm_' + msg + '.csv', index=False)

    def load_norm_normalizer_from_file(self, path, file):
        data = pd.read_csv(path + file, header=0).to_numpy()
        self.current_state_norm.running_ms.n = data[0, 0]
        self.current_state_norm.running_ms.mean = data[:, 1]
        self.current_state_norm.running_ms.std = data[:, 2]
        self.current_state_norm.running_ms.S = data[:, 3]
        self.next_state_norm.running_ms.n = data[0, 4]
        self.next_state_norm.running_ms.mean = data[:, 5]
        self.next_state_norm.running_ms.std = data[:, 6]
        self.next_state_norm.running_ms.S = data[:, 7]
