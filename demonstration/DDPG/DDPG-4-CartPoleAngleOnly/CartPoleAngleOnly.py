import numpy as np

from utils.functions import *
from algorithm.rl_base import rl_base
import cv2 as cv
from environment.color import Color
import pandas as pd


class CartPoleAngleOnly(rl_base):
    def __init__(self, initTheta: float):
        """
        :param initTheta:       initial angle, which should be less than 30 degree
        :param save_cfg:        save the model config file or not
        """
        super(CartPoleAngleOnly, self).__init__()
        '''physical parameters'''
        self.name = 'CartPoleAngleOnly'
        self.initTheta = initTheta
        self.theta = self.initTheta
        self.x = 0
        self.dtheta = 0.  # 从左往右转为正
        self.dx = 0.  # 水平向左为正
        self.force = 0.  # 外力，水平向左为正

        self.thetaMax = deg2rad(45)  # maximum angle
        # self.dthetaMax = deg2rad(720)   # maximum angular rate

        self.staticGain = 2.0
        self.norm_4_boundless_state = 4

        self.M = 1.0  # mass of the cart
        self.m = 0.1  # mass of the pole
        self.g = 9.8
        self.ell = 0.2  # 1 / 2 length of the pole
        self.kf = 0.2  # friction coefficient
        self.fm = 8  # maximum force added on the cart

        self.dt = 0.01  # 10ms
        self.timeMax = 6  # maximum time of each episode
        self.time = 0.
        self.etheta = 0. - self.theta
        self.ex = 0. - self.x
        '''physical parameters'''

        '''RL_BASE'''
        self.use_norm = True
        self.state_dim = 2  # theta, dtheta
        self.state_num = [np.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain],
                            [-np.inf, np.inf]]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.action_dim = 1
        self.action_step = [None]
        self.action_range = np.array([[-self.fm, self.fm]])
        self.action_num = [np.inf]
        self.action_space = [None]
        self.isActionContinuous = True
        self.initial_action = [self.force]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.Q_theta = 3.5  # cost for angular error
        self.Q_dtheta = 0.01  # cost for angular rate error
        self.R = 0.2  # cost for control input
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

        '''visualization_opencv'''
        self.width = 400
        self.height = 200
        self.name4image = 'CartPoleAngleOnly'
        self.xoffset = 0  # pixel
        self.scale = (self.width - 2 * self.xoffset) / 2 / 1.5  # m -> pixel
        self.cart_x_pixel = 40  # 仅仅为了显示，比例尺不一样的
        self.cart_y_pixel = 30
        self.pixel_per_n = 20  # 每牛顿的长度
        self.pole_ell_pixel = 50
        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        self.draw_init_image()
        '''visualization_opencv'''

    def draw_slide(self):
        pt1 = (self.xoffset, int(self.height / 2) - 1)
        pt2 = (self.width - 1 - self.xoffset, int(self.height / 2) + 1)
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=-1)

    def draw_cartpole_force(self):
        cx = self.xoffset + 1.5 * self.scale
        cy = self.height / 2
        pt1 = (int(cx - self.cart_x_pixel / 2), int(cy + self.cart_y_pixel / 2))
        pt2 = (int(cx + self.cart_x_pixel / 2), int(cy - self.cart_y_pixel / 2))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Orange, thickness=-1)

        pt1 = np.atleast_1d([int(cx), int(cy - self.cart_y_pixel / 2)])
        pt2 = np.atleast_1d([int(cx + self.pole_ell_pixel * S(self.theta)),
                             int(cy - self.cart_y_pixel / 2 - self.pole_ell_pixel * C(self.theta))])
        cv.line(img=self.image, pt1=pt1, pt2=pt2, color=Color().Red, thickness=4)
        if self.force >= 0:
            pt1 = np.atleast_1d([int(cx - self.cart_x_pixel / 2 - np.fabs(self.force) * self.pixel_per_n), int(cy)])
            pt2 = np.atleast_1d([int(cx - self.cart_x_pixel / 2), int(cy)])
        else:
            pt1 = np.atleast_1d([int(cx + self.cart_x_pixel / 2 + np.fabs(self.force) * self.pixel_per_n), int(cy)])
            pt2 = np.atleast_1d([int(cx + self.cart_x_pixel / 2), int(cy)])
        if np.fabs(self.force) > 1e-2:
            cv.arrowedLine(self.image, pt1, pt2, Color().Red, 2, 8, 0, 0.5 / np.fabs(self.force))

    def draw_center(self):
        cv.circle(self.image, (int(self.xoffset + 1.5 * self.scale), int(self.height / 2)), 4, Color().Black, -1)

    def make_text(self):
        # self.image = self.show.copy()
        cv.putText(self.image, "time : %.2f s" % self.time, (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Black, 2)
        cv.putText(self.image, "theta: %.3f " % (rad2deg(self.theta)), (20, 40), cv.FONT_HERSHEY_COMPLEX, 0.5,
                   Color().Black, 2)
        cv.putText(self.image, "  x  : %.3f m" % self.x, (20, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Black, 2)
        cv.putText(self.image, "force: %.3f N" % self.force, (20, 80), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Black, 2)

    def draw_init_image(self):
        # self.draw_slide()
        self.image_copy = self.image.copy()

    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_cartpole_force()
        self.make_text()
        self.draw_center()
        cv.imshow(self.name4image, self.image)
        cv.waitKey(1)

    def is_success(self):
        if np.linalg.norm([self.etheta, self.dtheta]) < 1e-2:
            return True
        return False

    def is_Terminal(self, param=None):
        """
        :brief:     判断回合是否结束
        :return:    是否结束
        """
        self.terminal_flag = 0
        if (self.theta > self.thetaMax + deg2rad(1)) or self.theta < -self.thetaMax - deg2rad(1):
            self.terminal_flag = 1
            # print('Angle out...')
            return True

        if self.time > self.timeMax:
            self.terminal_flag = 3
            print('Time out')
            return True

        # if self.is_success():
        # 	self.terminal_flag = 4
        # 	print('Success')
        # 	return True

        self.terminal_flag = 0
        return False

    def get_state(self):
        if self.use_norm:
            theta = self.theta / self.thetaMax
            dtheta = self.dtheta / self.norm_4_boundless_state
            s = np.array([theta, dtheta]) * self.staticGain
        else:
            s = np.array([self.theta, self.dtheta])
        return s

    def get_reward(self, param=None):
        """
        :param param:   extra parameters for reward function
        :return:
        """
        theta_middle = self.thetaMax / 2
        r1 = (theta_middle - np.fabs(self.theta)) * 10

        # r3 = 0
        r4 = 0.
        if self.terminal_flag == 1:		# 如果角度出界
            _n = (self.timeMax - self.time) / self.dt
            r4 = _n * r1
        # print('结束',_n, r1, r2, r3, r4)
        # print(self.theta)
        self.reward = r1 + r4

    def ode(self, xx: np.ndarray):
        """
        :param xx:  微分方程的状态，不是强化学习的状态。
        :return:
        """
        '''微分方程里面的状态：[theta, dtheta, x, dx]'''
        _theta = xx[0]
        _dtheta = xx[1]
        _x = xx[2]
        _dx = xx[3]
        ddx = (self.force +
               self.m * self.ell * _dtheta ** 2 * S(_theta)
               - self.kf * _dx
               - 3 / 4 * self.m * self.g * S(_theta) * C(_theta)) / \
              (self.M + self.m - 3 / 4 * self.m * C(_theta) ** 2)
        ddtheta = 3 / 4 / self.m / self.ell * (self.m * self.g * S(_theta) - self.m * ddx * C(_theta))
        dx = _dx
        dtheta = _dtheta

        return np.array([dtheta, ddtheta, dx, ddx])

    def rk44(self, action: np.ndarray):
        [self.force] = action
        h = self.dt / 10  # RK-44 解算步长
        tt = self.time + self.dt
        xx = np.array([self.theta, self.dtheta, self.x, self.dx])
        while self.time < tt:
            temp = self.ode(xx)
            K1 = h * temp
            K2 = h * self.ode(xx + K1 / 2)
            K3 = h * self.ode(xx + K2 / 2)
            K4 = h * self.ode(xx + K3)
            xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            self.time += h
        [self.theta, self.dtheta, self.x, self.dx] = xx.tolist()
        self.etheta = 0. - self.theta
        self.ex = 0. - self.x

    def step_update(self, action: list):
        self.force = action[0]  # get the extra force
        self.current_action = action.copy()
        self.current_state = self.get_state()
        self.rk44(np.array([self.force]))
        self.is_terminal = self.is_Terminal()
        self.next_state = self.get_state()
        self.get_reward()

    def reset(self, random: bool = False):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        if random:
            self.initTheta = np.random.uniform(-self.thetaMax / 2, self.thetaMax / 2)
        self.theta = self.initTheta
        self.x = 0
        self.dtheta = 0.  # 从左往右转为正
        self.dx = 0.  # 水平向左为正
        self.force = 0.  # 外力，水平向左为正
        self.time = 0.
        self.etheta = 0. - self.theta
        self.ex = 0. - self.x
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.initial_action = [self.force]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.draw_init_image()
