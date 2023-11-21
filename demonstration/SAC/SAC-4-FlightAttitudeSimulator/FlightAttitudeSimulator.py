import cv2 as cv

from utils.functions import *
from algorithm.rl_base import rl_base
from environment.color import Color


class FlightAttitudeSimulator(rl_base):
    def __init__(self, initTheta: float=0.):
        super(FlightAttitudeSimulator, self).__init__()
        '''physical parameters'''
        self.name = 'Flight_Attitude_Simulator'
        self.initTheta = initTheta
        self.setTheta = 0
        self.force = 0
        self.f_max = 4
        self.f_min = -1.5

        self.maxTheta = deg2rad(60.0)
        self.maxOmega = deg2rad(90.0)

        self.theta = self.initTheta
        self.thetaError = self.setTheta - self.theta
        self.dTheta = 0.0

        self.dt = 0.02  # control period
        self.time = 0.0

        self.sum_thetaError = 0.0
        self.timeMax = 5

        self.Lw = 0.02  # 杆宽度
        self.L = 0.362  # 杆半长
        self.J = 0.082  # 转动惯量
        self.k = 0.09  # 摩擦系数
        self.m = 0.3  # 配重重量
        self.dis = 0.3  # 铜块中心距中心距离0.059
        self.copperl = 0.06  # 铜块长度
        self.copperw = 0.03  # 铜块宽度
        self.g = 9.8  # 重力加速度
        '''physical parameters'''

        '''RL_BASE'''
        self.use_norm = True
        self.staticGain = 2
        self.state_dim = 2  # Theta, dTheta
        self.state_num = [np.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.maxTheta, self.maxTheta], [-self.maxOmega, self.maxOmega]]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.initial_state = self.get_state()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 1
        self.action_step = [None]
        self.action_range = np.array([[self.f_min, self.f_max]])
        self.action_num = [np.inf]
        self.action_space = [None]
        self.isActionContinuous = True
        self.current_action = np.array([0.0])

        self.reward = 0.0
        self.Q = 1
        self.Qv = 0.
        self.R = 0.1
        self.terminal_flag = 0  # 0-正常 1-上边界出界 2-下边界出界 3-超时
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

        self.draw_init_image()
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

    def draw_init_image(self):
        self.draw_base()
        self.image_copy = self.image.copy()

    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_pendulum()
        self.draw_copper()
        cv.imshow(self.name4image, self.image)
        cv.waitKey(1)

    def get_state(self):
        if self.use_norm:
            s = np.array([self.theta / self.maxTheta, self.dTheta / self.maxOmega]) * self.staticGain
        else:
            s = np.array([self.theta, self.dTheta])
        return s

    def is_success(self):
        if np.fabs(self.thetaError) < deg2rad(1):       # 角度误差小于1度
            if np.fabs(self.dTheta) < deg2rad(1):       # 速度也很小
                return True
        return False

    def is_Terminal(self, param=None):
        """
        :brief:     判断回合是否结束
        :return:    是否结束
        """
        self.terminal_flag = 0
        self.is_terminal = False
        if self.theta > self.maxTheta + deg2rad(1):
            self.terminal_flag = 1
            # print('超出最大角度')
            self.is_terminal = True
        if self.theta < -self.maxTheta - deg2rad(1):
            self.terminal_flag = 2
            # print('超出最小角度')
            self.is_terminal = True
        if self.time > self.timeMax:
            self.terminal_flag = 3
            print('Timeout')
            self.is_terminal = True
        # if self.is_success():
        #     self.terminal_flag = 4
        #     print('Success')
        #     return True

    def get_reward(self, param=None):
        Q = 3.
        R = 0.1

        r1 = -self.theta ** 2 * Q
        r2 = -self.dTheta ** 2 * R

        # r3 = 0.
        if self.terminal_flag == 1 or self.terminal_flag == 2:  # 出界
            _n = (self.timeMax - self.time) / self.dt
            r3 = _n * (r1 + r2)
        else:
            r3 = 0.
        self.reward = r1 + r2 + r3

    def ode(self, xx: np.ndarray):
        _dtheta = xx[1]
        _ddtheta = (self.force * self.L - self.m * self.g * self.dis - self.k * xx[1]) / (self.J + self.m * self.dis ** 2)
        return np.array([_dtheta, _ddtheta])

    def rk44(self, action: float):
        self.force = action
        xx_old = np.array([self.theta, self.dTheta])
        K1 = self.dt * self.ode(xx_old)
        K2 = self.dt * self.ode(xx_old + K1 / 2)
        K3 = self.dt * self.ode(xx_old + K2 / 2)
        K4 = self.dt * self.ode(xx_old + K3)
        xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        self.theta = xx_new[0]
        self.dTheta = xx_new[1]
        self.time += self.dt

    def step_update(self, action: np.ndarray):
        self.current_action = action.copy()
        self.current_state = self.get_state()
        self.rk44(action=action[0])
        self.is_Terminal()
        self.thetaError = self.setTheta - self.theta
        self.next_state = self.get_state()
        self.get_reward()

    def reset(self, random: bool = True):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        if random:
            self.initTheta = np.random.uniform(-self.maxTheta, self.maxTheta)
        self.theta = self.initTheta
        self.dTheta = 0.0
        self.time = 0.0
        self.thetaError = self.setTheta - self.theta
        self.sum_thetaError = 0.0
        self.image = np.ones([self.width, self.height, 3], np.uint8) * 255
        self.draw_init_image()
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.get_state()
        self.next_state = self.initial_state.copy()
        self.current_action = np.array([0.])
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''
