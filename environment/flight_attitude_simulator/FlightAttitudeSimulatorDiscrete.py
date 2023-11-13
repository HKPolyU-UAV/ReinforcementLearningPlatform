import numpy as np
import cv2 as cv

from utils.functions import *
from algorithm.rl_base import rl_base
from environment.color import Color


class FlightAttitudeSimulatorDiscrete(rl_base):
    def __init__(self, initTheta: float = 0.):
        """
        :brief:                 initialization
        :param initTheta:       initial theta
        """
        super(FlightAttitudeSimulatorDiscrete, self).__init__()
        self.name = 'FlightAttitudeSimulatorDiscrete'

        '''physical parameters'''
        self.initTheta = deg2rad(initTheta)
        self.f_max = 3.0
        self.f_min = -1.6
        self.f_step = 0.1
        self.theta_max = deg2rad(60.0)
        self.dtheta_max = deg2rad(90)
        self.static_gain = 2.0
        self.theta = self.initTheta
        self.dTheta = 0.0
        self.dt = 0.02
        self.time = 0.0
        self.timeMax = 5.0
        self.error = -self.theta
        self.J = 0.082  # 转动惯量
        self.k = 0.09  # 摩擦系数
        self.m = 0.3  # 配重重量
        self.g = 9.8  # 重力加速度
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.use_norm = True
        self.state_dim = 2  # error, dTheta,
        self.state_num = [np.inf, np.inf]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        if self.use_norm:
            self.state_range = [[-self.static_gain, self.static_gain],
                                [-self.static_gain, self.static_gain]]
        else:
            self.state_range = [[-self.theta_max, self.theta_max],
                                [-self.dtheta_max, self.dtheta_max]]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.initial_state = self.get_state()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 1
        self.action_step = [self.f_step]
        self.action_range = [[self.f_min, self.f_max]]
        self.action_num = [int((self.f_max - self.f_min) / self.f_step + 1)]
        self.action_space = [[self.f_min + i * self.f_step for i in range(self.action_num[0])]]
        self.isActionContinuous = False
        self.initial_action = [0.0]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''visualization_opencv'''
        self.width = 400
        self.height = 400
        self.image = np.ones([self.width, self.height, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        self.name4image = 'Flight attitude simulator'
        self.scale = 250  # cm -> pixel
        self.ybias = 360  # pixel
        self.base_hor_w = 0.4
        self.base_hor_h = 0.02
        self.base_ver_w = 0.02
        self.base_ver_h = 0.8
        self.dis = 0.3  # 铜块中心距中心距离0.059
        self.copperl = 0.06  # 铜块长度
        self.copperw = 0.03  # 铜块宽度
        self.Lw = 0.02  # 杆宽度
        self.L = 0.362  # 杆半长
        self.draw_base()
        '''visualization_opencv'''

    def draw_base(self):
        """
        :brief:     绘制基座
        :return:    None
        """
        pt1 = (int(self.width / 2 - self.base_hor_w * self.scale / 2), self.ybias)
        pt2 = (int(pt1[0] + self.base_hor_w * self.scale), int(pt1[1] - self.base_hor_h * self.scale))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=-1)
        pt1 = (int(self.width / 2 - self.base_ver_w * self.scale / 2), pt2[1])
        pt2 = (int(pt1[0] + self.base_ver_w * self.scale), int(pt2[1] - self.base_ver_h * self.scale))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=-1)
        self.image_copy = self.image.copy()

    def draw_pendulum(self):
        """
        :brief:     绘制摆杆
        :return:    None
        """
        cx = int(self.width / 2)
        cy = int(self.ybias - (self.base_hor_h + self.base_ver_h) * self.scale)
        theta1 = np.arctan(self.Lw / self.L / 2)
        theta2 = -theta1
        theta3 = np.pi + theta1
        theta4 = np.pi + theta2
        L0 = np.sqrt((self.Lw / 2) ** 2 + self.L ** 2)
        pt1 = np.atleast_1d([int(L0 * np.cos(theta1 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta1 + self.theta) * self.scale)])
        pt2 = np.atleast_1d([int(L0 * np.cos(theta2 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta2 + self.theta) * self.scale)])
        pt3 = np.atleast_1d([int(L0 * np.cos(theta3 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta3 + self.theta) * self.scale)])
        pt4 = np.atleast_1d([int(L0 * np.cos(theta4 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta4 + self.theta) * self.scale)])
        cv.fillPoly(img=self.image, pts=np.array([[pt1, pt2, pt3, pt4]]), color=Color().Red)

    def draw_copper(self):
        """
        :brief:     绘制铜块
        :return:    None
        """
        cx = int(self.width / 2)
        cy = int(self.ybias - (self.base_hor_h + self.base_ver_h) * self.scale)
        theta1 = np.arctan(self.copperw / 2 / (self.dis - self.copperl / 2))
        theta2 = np.arctan(self.copperw / 2 / (self.dis + self.copperl / 2))
        theta3 = -theta2
        theta4 = -theta1

        l1 = np.sqrt((self.copperw / 2) ** 2 + (self.dis - self.copperl / 2) ** 2)
        l2 = np.sqrt((self.copperw / 2) ** 2 + (self.dis + self.copperl / 2) ** 2)

        pt1 = np.atleast_1d([int(l1 * np.cos(theta1 + self.theta) * self.scale + cx),
                             int(cy - l1 * np.sin(theta1 + self.theta) * self.scale)])
        pt2 = np.atleast_1d([int(l2 * np.cos(theta2 + self.theta) * self.scale + cx),
                             int(cy - l2 * np.sin(theta2 + self.theta) * self.scale)])
        pt3 = np.atleast_1d([int(l2 * np.cos(theta3 + self.theta) * self.scale + cx),
                             int(cy - l2 * np.sin(theta3 + self.theta) * self.scale)])
        pt4 = np.atleast_1d([int(l1 * np.cos(theta4 + self.theta) * self.scale + cx),
                             int(cy - l1 * np.sin(theta4 + self.theta) * self.scale)])

        cv.fillPoly(img=self.image, pts=np.array([[pt1, pt2, pt3, pt4]]), color=Color().Black)

    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_pendulum()
        self.draw_copper()
        cv.imshow(self.name4image, self.image)
        cv.waitKey(1)

    def get_state(self):
        if self.use_norm:
            return np.array([-self.theta / self.theta_max, self.dTheta / self.dtheta_max]) * self.static_gain
        else:
            return np.array([-self.theta, self.dTheta])

    def is_success(self):
        b1 = np.fabs(self.theta) < deg2rad(1)
        b2 = np.fabs(self.dTheta) < deg2rad(1)
        return b1 and b2

    def is_out(self):
        if self.theta > self.theta_max + deg2rad(1):
            # self.terminal_flag = 1
            # print('超出最大角度')
            return True
        if self.theta < -self.theta_max - deg2rad(1):
            # self.terminal_flag = 1
            # print('超出最小角度')
            return True

    def is_Terminal(self, param=None):
        """
        :brief:     判断回合是否结束
        :return:    是否结束
        """
        self.terminal_flag = 0
        self.is_terminal = False
        if self.is_out():
            self.terminal_flag = 1
            return True
        if self.time > self.timeMax:
            self.terminal_flag = 2
            print('超时')
            return True
        # if self.is_success():
        #     self.terminal_flag = 3
        #     print('成功')
        #     return True

    def step_update(self, action:np.ndarray):
        self.current_action = action.copy()

        def f(angle, dangle):
            a2 = -self.k / (self.J + self.m * self.dis ** 2)
            a1 = -self.m * self.g * self.dis / (self.dis + self.m * self.dis ** 2)
            a0 = self.L * self.current_action[0] / (self.J + self.m * self.dis ** 2)
            return a2 * dangle + a1 * np.cos(angle) + a0

        h = self.dt / 1
        t_sim = 0.0
        self.current_state = self.get_state()
        '''differential equation'''
        while t_sim <= self.dt:
            K1 = self.dTheta
            L1 = f(self.theta, self.dTheta)
            K2 = self.dTheta + h * L1 / 2
            L2 = f(self.theta + h * K1 / 2, self.dTheta + h * L1 / 2)
            K3 = self.dTheta + h * L2 / 2
            L3 = f(self.theta + h * K2 / 2, self.dTheta + h * L2 / 2)
            K4 = self.dTheta + h * L3
            L4 = f(self.theta + h * K3, self.dTheta + h * L3)
            self.theta = self.theta + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6
            self.dTheta = self.dTheta + h * (L1 + 2 * L2 + 2 * L3 + L4) / 6
            t_sim = t_sim + h
        if self.theta > self.theta_max:
            self.theta = self.theta_max
            self.dTheta = -0.8 * self.dTheta
        if self.theta < -self.theta_max:
            self.theta = -self.theta_max
            self.dTheta = -0.8 * self.dTheta
        self.time = self.time + self.dt
        self.error = -self.theta
        self.next_state = self.get_state()
        '''differential equation'''

        self.is_Terminal()
        self.get_reward()

    def get_reward(self, param=None):
        Q = 3.
        R = 0.0

        r1 = -self.theta ** 2 * Q
        r2 = -self.dTheta ** 2 * R

        # r3 = 0.
        if self.terminal_flag == 1 or self.terminal_flag == 2:      # 出界
            _n = (self.timeMax - self.time) / self.dt
            r3 = _n * (r1 + r2)
        else:
            r3 = 0.
        self.reward = r1 + r2 + r3

    def reset(self):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        self.theta = self.initTheta
        self.dTheta = 0.0
        self.time = 0.0
        self.error = -self.theta
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

    def reset_random(self):
        """
        :brief:
        :return:
        """
        '''physical parameters'''
        self.initTheta = np.random.uniform(-self.theta_max, self.theta_max)
        self.theta = self.initTheta
        self.dTheta = 0.0
        self.time = 0.0
        self.error = -self.theta
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.initial_state = self.get_state()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''
