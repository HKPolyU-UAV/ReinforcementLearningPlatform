import math
import os
import sys
from numpy import deg2rad

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')

from algorithm.rl_base import rl_base
from environment.uav_robust.Color import Color
from environment.uav_robust.FNTSMC import fntsmc_att, fntsmc_param
from environment.uav_robust.collector import data_collector
from environment.uav_robust.ref_cmd import *
from environment.uav_robust.uav import UAV, uav_param
from environment.uav_robust.uav_pos_ctrl import uav_pos_ctrl


class uav_hover_outer_loop(rl_base, uav_pos_ctrl):
    def __init__(self, UAV_param: uav_param, pos_ctrl_param: fntsmc_param,
                 att_ctrl_param: fntsmc_param, target0: np.ndarray):
        rl_base.__init__(self)
        uav_pos_ctrl.__init__(self, UAV_param, att_ctrl_param, pos_ctrl_param)

        self.uav_param = UAV_param
        self.name = 'uav_hover_outer_loop'

        self.collector = data_collector(round(self.time_max / self.dt))

        self.pos_ref = target0
        self.error = self.uav_pos() - self.pos_ref
        self.init_image()

        '''state action limitation'''
        self.static_gain = 1.0

        self.e_pos_max = np.array([5., 5., 5.])
        self.e_pos_min = -np.array([5., 5., 0.])
        self.vel_max = np.array([3., 3., 3.])
        self.vel_min = -np.array([3., 3., 3.])
        self.dot_att_min = np.array([-deg2rad(60), -deg2rad(60), -deg2rad(1)])
        self.dot_att_max = np.array([deg2rad(60), deg2rad(60), deg2rad(1)])
        self.u_min = -8
        self.u_max = 8
        '''state action limitation'''

        '''rl_base'''
        self.state_dim = 6  # ex ey ez vx vy vz
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.static_gain, self.static_gain] for _ in range(self.state_dim)]
        self.is_state_continuous = [True for _ in range(self.state_dim)]

        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 3  # ux uy uz
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.action_range = [[self.u_min, self.u_max],
                             [self.u_min, self.u_max],
                             [self.u_min, self.u_max]]
        self.is_action_continuous = [True for _ in range(self.action_dim)]

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞
        '''rl_base'''

    def state_norm(self) -> np.ndarray:
        """
        RL状态归一化
        """
        norm_error = self.error / (self.e_pos_max - self.e_pos_min) * self.static_gain
        norm_vel = 2 * self.uav_vel() / (self.vel_max - self.vel_min) * self.static_gain
        norm_state = np.concatenate((norm_error, norm_vel))

        return norm_state

    def get_reward(self, param=None):
        """
        计算奖励
        """
        Qx, Qv, R = 1, 0.1, 0.02
        r1 = - np.linalg.norm(np.tanh(10 * self.error)) ** 2 * 0.5 * Qx - np.linalg.norm(self.error) ** 2 * 0.5 * Qx
        r2 = - np.linalg.norm(np.tanh(10 * self.uav_vel())) ** 2 * 0.5 * Qx - np.linalg.norm(
            self.uav_vel()) ** 2 * 0.5 * Qv
        r3 = - np.linalg.norm(self.current_action) ** 2 * R

        r4 = 0
        # 如果因为越界终止，则给剩余时间可能取得的最大惩罚
        if self.is_pos_out() or self.is_att_out():
            r4 = - (self.time_max - self.time) / self.dt * (Qx * np.linalg.norm(self.error) ** 2
                                                            + Qv * np.linalg.norm(self.uav_vel()) ** 2
                                                            + R * np.linalg.norm(self.current_action) ** 2)
        self.reward = r1 + r2 + r3 + r4

    def is_Terminal(self, param=None):
        self.is_terminal, self.terminal_flag = self.is_episode_Terminal()

    def step_update(self, action: np.ndarray):
        """
        @param action:  三轴加速度指令 ux uy uz
        @return:
        """
        self.current_action = action.copy()
        self.current_state = self.state_norm()

        # 外环由RL控制给出
        self.pos_ctrl.control = action.copy()
        phi_d, theta_d, uf = self.uo_2_ref_angle_throttle()
        phi_d = np.clip(phi_d, self.att_zone[0][0], self.att_zone[0][1])
        theta_d = np.clip(theta_d, self.att_zone[1][0], self.att_zone[1][1])

        # 计算内环控制所需参数
        att_ref_old = self.att_ref
        att_ref = np.array([phi_d, theta_d, 0.0])  # 偏航角手动设置为0
        self.dot_att_ref = (att_ref - att_ref_old) / self.dt
        self.dot_att_ref = np.clip(self.dot_att_ref, self.dot_att_min, self.dot_att_max)  # 姿态角指令变化率限制，保证内环跟得上
        self.att_ref = self.dot_att_ref * self.dt + att_ref_old

        # 内环由FNTSMC给出
        torque = self.att_control(self.att_ref, self.dot_att_ref, np.zeros(3), att_only=False)  # 内环fntsmc控制

        # 合并成总控制量：油门 + 三个转矩
        a = np.concatenate(([uf], torque))  # 真实控制量

        self.update(action=a)
        self.error = self.uav_pos() - self.pos_ref

        self.is_Terminal()
        self.next_state = self.state_norm()

        self.get_reward()

    def reset(self):
        self.m = self.param.m
        self.g = self.param.g
        self.J = self.param.J
        self.d = self.param.d
        self.CT = self.param.CT
        self.CM = self.param.CM
        self.J0 = self.param.J0
        self.kr = self.param.kr
        self.kt = self.param.kt

        self.x = self.param.pos0[0]
        self.y = self.param.pos0[1]
        self.z = self.param.pos0[2]
        self.vx = self.param.vel0[0]
        self.vy = self.param.vel0[1]
        self.vz = self.param.vel0[2]
        self.phi = self.param.angle0[0]
        self.theta = self.param.angle0[1]
        self.psi = self.param.angle0[2]
        self.p = self.param.pqr0[0]
        self.q = self.param.pqr0[1]
        self.r = self.param.pqr0[2]

        self.dt = self.param.dt
        self.n = 0  # 记录走过的拍数
        self.time = 0.  # 当前时间
        self.time_max = self.param.time_max

        self.throttle = self.m * self.g  # 油门
        self.torque = np.array([0., 0., 0.]).astype(float)  # 转矩
        self.terminal_flag = 0

        self.pos_zone = self.param.pos_zone
        self.att_zone = self.param.att_zone
        self.x_min = self.pos_zone[0][0]
        self.x_max = self.pos_zone[0][1]
        self.y_min = self.pos_zone[1][0]
        self.y_max = self.pos_zone[1][1]
        self.z_min = self.pos_zone[2][0]
        self.z_max = self.pos_zone[2][1]

        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        self.init_image()

        self.error = self.uav_pos() - self.pos_ref
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.collector.reset(round(self.time_max / self.dt))
        self.reward = 0.0
        self.is_terminal = False

    def reset_random(self):
        """
        定点控制可以选择起始点和目标点随机
        """
        self.reset()
        self.pos_ref = self.generate_random_point(offset=1.0)  # 随即目标点
        # self.x, self.y, self.z = self.generate_random_point(offset=1.0)  # 随即初始位置

    def generate_random_point(self, offset: float):
        """
        在飞行范围内随即选择一个点，offset防止过于贴近边界
        """
        return np.random.uniform(low=self.pos_zone[:, 0] + offset, high=self.pos_zone[:, 1] - offset)

    def init_image(self):
        self.draw_init_image()

    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_3d_points_projection(np.atleast_2d([self.uav_pos()]), [Color().Red])
        self.draw_3d_points_projection(np.atleast_2d([self.pos_ref[0:3]]), [Color().Green])
        self.draw_error(self.uav_pos(), self.pos_ref[0:3])
        self.show_image(iswait=False)
