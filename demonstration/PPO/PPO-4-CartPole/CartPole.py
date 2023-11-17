import math

import numpy as np

from utils.functions import *
from algorithm.rl_base import rl_base
import cv2 as cv
from environment.color import Color
import pandas as pd


class CartPole(rl_base):
    def __init__(self, initTheta: float, initX: float, save_cfg: bool = True):
        """
        :param initTheta:       initial angle, which should be less than 30 degree
        :param initX:           initial position
        :param save_cfg:        save the model config file or not
        """
        super(CartPole, self).__init__()
        '''physical parameters'''
        self.initTheta = initTheta
        self.initX = initX
        self.theta = self.initTheta
        self.x = self.initX
        self.dtheta = 0.  # 从左往右转为正
        self.dx = 0.  # 水平向左为正
        self.force = 0.  # 外力，水平向左为正

        self.thetaMax = deg2rad(45)  # maximum angle
        # self.dthetaMax = deg2rad(720)   # maximum angular rate
        self.xMax = 1.5  # maximum distance
        # self.dxMax = 10                 # maximum velicity

        self.staticGain = 2.0
        self.norm_4_boundless_state = 1
        # 有一些变量本身在物理系统中不做限制，但是为了防止在训练时变量数值差距太大
        # 所以将该变量除以norm_4_boundless_state，再乘以staticGain，再送入神经网络

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
        self.name = 'CartPole'
        '''physical parameters'''

        '''RL_BASE'''
        self.state_dim = 4  # theta, dtheta, x, dx
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.use_norm = True
        if self.use_norm:
            self.state_range = [[-self.staticGain, self.staticGain],
                                [-self.staticGain, self.staticGain],
                                [-self.staticGain, self.staticGain],
                                [-self.staticGain, self.staticGain]]
        else:
            self.state_range = [[-self.thetaMax, self.thetaMax],
                                [-np.inf, np.inf],
                                [-self.xMax, self.xMax],
                                [-np.inf, np.inf]]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.action_dim = 1
        self.action_step = [None]
        self.action_range = np.array([[-self.fm, self.fm]])
        self.action_num = [math.inf]
        self.action_space = [None]
        self.isActionContinuous = True
        self.current_action = np.zeros(self.action_dim)

        self.reward = 0.0
        self.Q_x = 100  # cost for position error
        self.Q_dx = 0.2  # cost for linear velocity error
        self.Q_theta = 200  # cost for angular error
        self.Q_dtheta = 0.1  # cost for angular rate error
        self.R = 0.5  # cost for control input
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

        '''visualization_opencv'''
        self.width = 400
        self.height = 200
        self.image = np.zeros([self.height, self.width, 3], np.uint8)
        self.image[:, :, 0] = np.ones([self.height, self.width]) * 255
        self.image[:, :, 1] = np.ones([self.height, self.width]) * 255
        self.image[:, :, 2] = np.ones([self.height, self.width]) * 255
        self.name4image = 'cartpole'
        self.xoffset = 0  # pixel
        self.scale = (self.width - 2 * self.xoffset) / 2 / self.xMax  # m -> pixel
        self.cart_x_pixel = 40  # 仅仅为了显示，比例尺不一样的
        self.cart_y_pixel = 30
        self.pixel_per_n = 20  # 每牛顿的长度
        self.pole_ell_pixel = 50

        self.show = self.image.copy()
        self.save = self.image.copy()
        '''visualization_opencv'''

        '''datasave'''
        self.save_X = [self.x]
        self.save_Theta = [self.theta]
        self.save_dX = [self.dx]
        self.save_dTheta = [self.dtheta]
        self.save_ex = [self.ex]
        self.save_eTheta = [self.etheta]
        self.saveTime = [self.time]
        '''datasave'''

    def draw_slide(self):
        pt1 = (self.xoffset, int(self.height / 2) - 1)
        pt2 = (self.width - 1 - self.xoffset, int(self.height / 2) + 1)
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=-1)
        self.show = self.image.copy()  # show是基础画布

    def draw_cartpole(self):
        # self.image = self.show.copy()
        cx = self.xoffset + (self.x + self.xMax) * self.scale
        cy = self.height / 2
        pt1 = (int(cx - self.cart_x_pixel / 2), int(cy + self.cart_y_pixel / 2))
        pt2 = (int(cx + self.cart_x_pixel / 2), int(cy - self.cart_y_pixel / 2))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Orange, thickness=-1)

        pt1 = np.atleast_1d([int(cx), int(cy - self.cart_y_pixel / 2)])
        pt2 = np.atleast_1d([int(cx + self.pole_ell_pixel * math.sin(self.theta)),
                             int(cy - self.cart_y_pixel / 2 - self.pole_ell_pixel * math.cos(self.theta))])
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
        cv.circle(self.image, (int(self.xoffset + self.xMax * self.scale), int(self.height / 2)), 4, Color().Black, -1)

    def make_text(self):
        # self.image = self.show.copy()
        cv.putText(self.image, "time : %.2f s" % self.time, (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.4, Color().Black, 1)
        cv.putText(self.image, "theta: %.3f " % (rad2deg(self.theta)), (20, 40), cv.FONT_HERSHEY_COMPLEX, 0.4,
                   Color().Black, 1)
        cv.putText(self.image, "  x  : %.3f m" % self.x, (20, 60), cv.FONT_HERSHEY_COMPLEX, 0.4, Color().Black, 1)

    def draw_init_image(self):
        self.draw_slide()

    def visualization(self):
        self.image = self.show.copy()
        self.draw_cartpole()
        self.make_text()
        self.draw_center()
        cv.imshow(self.name4image, self.image)
        cv.waitKey(1)

    def get_state(self):
        if self.use_norm:
            theta = self.theta / self.thetaMax * self.staticGain
            dtheta = self.dtheta / self.norm_4_boundless_state * self.staticGain
            x = self.x / self.xMax * self.staticGain
            dx = self.dx / self.norm_4_boundless_state * self.staticGain
            state = np.array([theta, dtheta, x, dx])
        else:
            state = np.array([self.theta, self.dtheta, self.x, self.dx])
        return state

    def is_success(self):
        if np.linalg.norm([self.ex, self.dx, self.etheta, self.dtheta]) < 1e-2:
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

        if self.x > self.xMax or self.x < -self.xMax:
            self.terminal_flag = 2
            # print('Position out...')
            return True

        if self.time > self.timeMax:
            self.terminal_flag = 3
            # print('Time out')
            return True

        if self.is_success():
            self.terminal_flag = 4
            # print('Success')
            return True

        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        """
        :param param:   extra parameters for reward function
        :return:
        """
        '''Should be a function with respec to [theta, dtheta, etheta, x, dx ,ex]'''
        '''
		The values of x and ex are identical in the env, and so are theta and etheta. 
		'''
        r1 = -self.Q_x * self.ex ** 2  # Qx = 1
        r2 = -self.Q_dx * self.dx ** 2  # Qdx = 0.1
        r3 = -self.Q_theta * self.etheta ** 2  # Qtheta = 200
        r4 = -self.Q_dtheta * self.dtheta ** 2  # Qdtheta = 0.1
        r5 = -self.R * self.force ** 2  # QR = 0.1
        if self.terminal_flag == 1:
            r6 = -100
        elif self.terminal_flag == 2:
            r6 = -100
        elif self.terminal_flag == 3:
            r6 = 0
        elif self.terminal_flag == 4:
            r6 = 200
        else:
            r6 = 0
        # self.reward = r1 + r2 + r3 + r4 + r5 + r6
        self.reward = -0.1 * (5 * self.etheta ** 2 + self.ex ** 2 + 0.05 * self.force ** 2) + r6
        '''玄学，完全是玄学, sun of a bitch'''

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
               self.m * self.ell * _dtheta ** 2 * math.sin(_theta)
               - self.kf * _dx
               - 3 / 4 * self.m * self.g * math.sin(_theta) * math.cos(_theta)) / \
              (self.M + self.m - 3 / 4 * self.m * math.cos(_theta) ** 2)
        ddtheta = 3 / 4 / self.m / self.ell * (self.m * self.g * math.sin(_theta) - self.m * ddx * math.cos(_theta))
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

    def step_update(self, action: list):
        self.force = action[0]  # get the extra force
        self.current_action = action.copy()
        self.etheta = 0. - self.theta
        self.ex = 0. - self.x
        self.current_state = self.get_state()
        '''RK-44'''
        self.rk44(np.array([self.force]))
        '''RK-44'''

        '''角度，位置误差更新'''
        self.etheta = 0. - self.theta
        self.ex = 0. - self.x
        self.is_terminal = self.is_Terminal()
        self.next_state = self.get_state()
        '''角度，位置误差更新'''
        self.get_reward()

    def reset(self, random: bool = False):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        if random:
            self.initTheta = np.random.uniform(-self.thetaMax / 2, self.thetaMax / 2)
            self.initX = np.random.uniform(-self.xMax / 2, self.xMax / 2)
        self.theta = self.initTheta
        self.x = self.initX
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

        self.current_action = np.zeros(self.action_dim)

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

        '''data_save'''
        self.save_X = [self.x]
        self.save_Theta = [self.theta]
        self.save_dX = [self.dx]
        self.save_dTheta = [self.dtheta]
        self.save_ex = [self.ex]
        self.save_eTheta = [self.etheta]
        self.saveTime = [self.time]
        '''data_save'''

    def reset_with_para(self, para=None):
        """
        :param para:    两个参数 = [initTheta initX]
        :return:
        """
        '''physical parameters'''
        self.initTheta = para[0]
        self.initX = para[1]
        self.theta = self.initTheta
        self.x = self.initX
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

        self.current_action = np.zeros(self.action_dim)

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

        '''data_save'''
        self.save_X = [self.x]
        self.save_Theta = [self.theta]
        self.save_dX = [self.dx]
        self.save_dTheta = [self.dtheta]
        self.save_ex = [self.ex]
        self.save_eTheta = [self.etheta]
        self.saveTime = [self.time]
        '''data_save'''

    def saveData(self, is2file=False, filename='cartpole.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'x': self.save_X,
                'theta': self.save_Theta,
                'dx': self.save_dX,
                'dtheta': self.save_dTheta,
                'ex': self.save_ex,
                'etheta': self.save_eTheta,
                'time': self.saveTime
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_X.append(self.x)
            self.save_Theta.append(self.theta)
            self.save_dX.append(self.dx)
            self.save_dTheta.append(self.dtheta)
            self.save_ex.append(self.ex)
            self.save_eTheta.append(self.etheta)
            self.saveTime.append(self.time)
