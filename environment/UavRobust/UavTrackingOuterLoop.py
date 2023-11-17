import math
import os
import sys
from numpy import deg2rad
from algorithm.rl_base import rl_base
from environment.color import Color
from FNTSMC import fntsmc_param
from collector import data_collector
from ref_cmd import *
from uav import uav_param
from uav_pos_ctrl import uav_pos_ctrl

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../'))


class uav_tracking_outer_loop(rl_base, uav_pos_ctrl):
    def __init__(self, UAV_param: uav_param, pos_ctrl_param: fntsmc_param,
                 att_ctrl_param: fntsmc_param, ref_amplitude: np.ndarray,
                 ref_period: np.ndarray, ref_bias_a: np.ndarray, ref_bias_phase: np.ndarray):
        rl_base.__init__(self)
        uav_pos_ctrl.__init__(self, UAV_param, att_ctrl_param, pos_ctrl_param)

        self.uav_param = UAV_param
        self.name = 'uav_tracking_outer_loop'

        self.collector = data_collector(round(self.time_max / self.dt))

        '''Define parameters for signal generator'''
        self.ref_amplitude = ref_amplitude
        self.ref_period = ref_period
        self.ref_bias_a = ref_bias_a
        self.ref_bias_phase = ref_bias_phase
        self.trajectory = self.generate_ref_pos_trajectory(self.ref_amplitude, self.ref_period,
                                                           self.ref_bias_a, self.ref_bias_phase)
        '''Define parameters for signal generator'''

        self.pos_ref, self.dot_pos_ref, _, _ = ref_uav(self.time, self.ref_amplitude, self.ref_period, self.ref_bias_a,
                                                       self.ref_bias_phase)
        self.error = self.uav_pos() - self.pos_ref
        self.dot_error = self.uav_vel() - self.dot_pos_ref

        '''state action limitation'''
        self.static_gain = 1.0

        self.e_pos_max = np.array([5., 5., 5.])
        self.e_pos_min = -np.array([5., 5., 0.])
        self.e_vel_max = np.array([3., 3., 3.])
        self.e_vel_min = -np.array([3., 3., 3.])
        self.dot_att_min = np.array([-deg2rad(60), -deg2rad(60), -deg2rad(1)])
        self.dot_att_max = np.array([deg2rad(60), deg2rad(60), deg2rad(1)])
        self.u_min = -8
        self.u_max = 8
        '''state action limitation'''

        '''rl_base'''
        self.state_dim = 6  # ex ey ez evx evy evz
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.use_norm = True
        if self.use_norm:
            self.state_range = [[-self.static_gain, self.static_gain] for _ in range(self.state_dim)]
        else:
            self.state_range = [[self.e_pos_min[0], self.e_pos_max[0]],
                                [self.e_pos_min[1], self.e_pos_max[1]],
                                [self.e_pos_min[2], self.e_pos_max[2]],
                                [self.e_vel_min[0], self.e_vel_max[0]],
                                [self.e_vel_min[1], self.e_vel_max[1]],
                                [self.e_vel_min[2], self.e_vel_max[2]]]
        self.is_state_continuous = [True for _ in range(self.state_dim)]

        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.action_dim = 3  # ux uy uz
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.action_range = [[self.u_min, self.u_max],
                             [self.u_min, self.u_max],
                             [self.u_min, self.u_max]]
        self.is_action_continuous = [True for _ in range(self.action_dim)]
        self.current_action = np.zeros(self.action_dim)

        self.reward = 0.
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞
        '''rl_base'''

    def get_state(self) -> np.ndarray:
        """
        RL状态归一化
        """
        if self.use_norm:
            norm_error = self.error / (self.e_pos_max - self.e_pos_min) * self.static_gain
            norm_dot_error = self.dot_error / (self.e_vel_max - self.e_vel_min) * self.static_gain
            state = np.concatenate((norm_error, norm_dot_error))
        else:
            state = np.concatenate((self.error, self.dot_error))
        return state

    def get_reward(self, param=None):
        """
        计算奖励
        """
        Qx, Qv, R = 1, 0.1, 0.01
        r1 = - np.linalg.norm(self.error) ** 2 * Qx
        r2 = - np.linalg.norm(self.dot_error) ** 2 * Qv
        # robust reward function
        r1 -= np.linalg.norm(np.tanh(10 * self.error)) ** 2 * Qx
        r2 -= np.linalg.norm(np.tanh(10 * self.dot_error)) ** 2 * Qv
        r3 = - np.linalg.norm(self.current_action) ** 2 * R

        r4 = 0
        # 如果因为越界终止，则给剩余时间可能取得的最大惩罚
        if self.is_pos_out() or self.is_att_out():
            r4 = (self.time_max - self.time) / self.dt * (r1 + r2 + r3)
        self.reward = r1 + r2 + r3 + r4

    def is_Terminal(self, param=None):
        self.is_terminal, self.terminal_flag = self.is_episode_Terminal()

    def step_update(self, action: np.ndarray):
        """
        @param action:  三轴加速度指令 ux uy uz
        @return:
        """
        self.current_action = action.copy()
        self.current_state = self.get_state()

        # 外环由RL控制给出
        self.pos_ctrl.control = action.copy()
        phi_d, theta_d, uf = self.uo_2_ref_angle_throttle()
        phi_d = np.clip(phi_d, self.att_zone[0][0], self.att_zone[0][1])    # 姿态指令不能越界
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
        self.pos_ref, self.dot_pos_ref, _, _ = ref_uav(self.time, self.ref_amplitude, self.ref_period, self.ref_bias_a,
                                                       self.ref_bias_phase)
        self.error = self.uav_pos() - self.pos_ref
        self.dot_error = self.uav_vel() - self.dot_pos_ref

        self.is_Terminal()
        self.next_state = self.get_state()

        self.get_reward()

    def reset(self, random: bool = False):
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

        self.generate_random_trajectory(is_random=random, outer_param=None)
        if random:
            self.x, self.y, self.z = \
                self.set_random_init_pos(pos0=self.trajectory[0][0:3], r=0.3 * np.ones(3))  # 设置初始位置在轨迹第一个点附近
        self.draw_init_image()
        self.pos_ref, self.dot_pos_ref, _, _ = ref_uav(self.time, self.ref_amplitude, self.ref_period, self.ref_bias_a,
                                                       self.ref_bias_phase)
        self.error = self.uav_pos() - self.pos_ref
        self.dot_error = self.uav_vel() - self.dot_pos_ref
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.current_action = np.zeros(self.action_dim)

        self.collector.reset(round(self.time_max / self.dt))
        self.reward = 0.0
        self.is_terminal = False

    def generate_random_trajectory(self, is_random: bool = False, outer_param: list = None):
        """
        @param outer_param: 手动指定轨迹参数
        @param is_random:	随机在振幅与周期
        @return:			None
        """
        center = np.mean(self.pos_zone, axis=1)
        if outer_param is not None:
            A = outer_param[0]
            T = outer_param[1]
            phi0 = outer_param[2]
        else:
            if is_random:
                A = np.array([
                    np.random.uniform(low=0., high=self.x_max - center[0] - 1),
                    np.random.uniform(low=0., high=self.y_max - center[1] - 1),
                    np.random.uniform(low=0., high=self.z_max - center[2] - 1)
                ])
                T = np.random.uniform(low=5, high=10, size=3)  # 随机生成周期
                phi0 = np.random.uniform(low=0, high=np.pi / 2, size=3)
                # phi0 = np.array([0., 0., np.pi / 2])
            else:
                A = np.array([1.5, 1.5, 0.3])
                T = np.array([6., 6., 10])
                phi0 = np.array([np.pi / 2, 0., 0.])

        self.ref_amplitude = A
        self.ref_period = T
        self.ref_bias_a = center
        self.ref_bias_phase = phi0
        self.trajectory = self.generate_ref_pos_trajectory(self.ref_amplitude, self.ref_period,
                                                           self.ref_bias_a, self.ref_bias_phase)

    def draw_init_image(self):
        self.draw_3d_trajectory_projection(self.trajectory)
        self.draw_boundary()
        self.draw_label()
        self.draw_region_grid(6, 6, 6)
        self.draw_axis(6, 6, 6)
        self.image_copy = self.image.copy()

    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_3d_points_projection(np.atleast_2d([self.uav_pos()]), [Color().Red])
        self.draw_3d_points_projection(np.atleast_2d([self.pos_ref[0:3]]), [Color().Green])
        self.draw_error(self.uav_pos(), self.pos_ref[0:3])
        self.show_image(iswait=False)
