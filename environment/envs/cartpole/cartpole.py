import sys
import os

import numpy as np

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from common.common_func import *
from algorithm.rl_base.rl_base import rl_base
import cv2 as cv
from environment.Color import Color
from environment.config.xml_write import *
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
        self.state_range = [[-self.staticGain, self.staticGain],
                            [-math.inf, math.inf],
                            [-self.staticGain, self.staticGain],
                            [-math.inf, math.inf]]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.initial_state = np.array([self.theta / self.thetaMax * self.staticGain,
                                       self.dtheta / self.norm_4_boundless_state * self.staticGain,
                                       self.x / self.xMax * self.staticGain,
                                       self.dx / self.norm_4_boundless_state * self.staticGain])
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 1
        self.action_step = [None]
        self.action_range = [[-self.fm, self.fm]]
        self.action_num = [math.inf]
        self.action_space = [None]
        self.isActionContinuous = True
        self.initial_action = [self.force]
        self.current_action = self.initial_action.copy()

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

        # self.draw_slide()
        # self.draw_cartpole()
        # self.make_text()
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

        if save_cfg:
            self.saveModel2XML()

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

    def show_dynamic_image(self, isWait=False):
        self.image = self.show.copy()
        self.draw_cartpole()
        self.make_text()
        self.draw_center()
        cv.imshow(self.name4image, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)

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

    # if self.x > 0:			# x+
    # 	if self.theta < 0:	# x+ theta-
    # 		if self.dx < 0:	# x+ theta- dx- 对
    # 			r1 = 3
    # 		else:			# x+ theta- dx+ 错
    # 			r1 = -3
    # 	else:				# x+ theta+
    # 		if self.dx < 0:	# x+ theta+ dx- 错
    # 			r1 = -3
    # 		else:			# x+ theta+ dx+ 对
    # 			r1 = 3
    # else:					# x-
    # 	if self.theta < 0:	# x- theta-
    # 		if self.dx < 0:	# x- theta- dx- 对
    # 			r1 = 3
    # 		else:			# x- theta- dx+ 错
    # 			r1 = -3
    # 	else:				# x- theta+
    # 		if self.dx < 0:	# x- theta+ dx- 错
    # 			r1 = -3
    # 		else:			# x- theta+ dx+ 对
    # 			r1 = 3
    # r2 = self.force ** 2 * 0.1
    # if self.terminal_flag == 1:		# angle out
    # 	r3 = -20
    # elif self.terminal_flag == 2:	# position out
    # 	r3 = -10
    # elif self.terminal_flag == 3:	# time out
    # 	r3 = 10
    # elif self.terminal_flag == 4:	# success
    # 	r3 = 100
    # else:
    # 	r3 = 0
    # self.reward = r1 + r2 + r3

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
        self.current_state = np.array([self.theta / self.thetaMax * self.staticGain,
                                       self.dtheta / self.norm_4_boundless_state * self.staticGain,
                                       self.x / self.xMax * self.staticGain,
                                       self.dx / self.norm_4_boundless_state * self.staticGain])
        '''RK-44'''
        self.rk44(np.array([self.force]))
        '''RK-44'''

        '''角度，位置误差更新'''
        self.etheta = 0. - self.theta
        self.ex = 0. - self.x
        self.is_terminal = self.is_Terminal()
        self.next_state = np.array([self.theta / self.thetaMax * self.staticGain,
                                    self.dtheta / self.norm_4_boundless_state * self.staticGain,
                                    self.x / self.xMax * self.staticGain,
                                    self.dx / self.norm_4_boundless_state * self.staticGain])
        '''角度，位置误差更新'''
        self.get_reward()
        return self.current_state, action, self.reward, self.next_state, self.is_terminal

    def reset(self):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
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
        self.initial_state = np.array([self.theta / self.thetaMax * self.staticGain,
                                       self.dtheta / self.norm_4_boundless_state * self.staticGain,
                                       self.x / self.xMax * self.staticGain,
                                       self.dx / self.norm_4_boundless_state * self.staticGain])
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [self.force]
        self.current_action = self.initial_action.copy()

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

    def reset_random(self):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        self.initTheta = random.uniform(-self.thetaMax / 2, self.thetaMax / 2)
        self.initX = random.uniform(-self.xMax / 2, self.xMax / 2)
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
        self.initial_state = np.array([self.theta / self.thetaMax * self.staticGain,
                                       self.dtheta / self.norm_4_boundless_state * self.staticGain,
                                       self.x / self.xMax * self.staticGain,
                                       self.dx / self.norm_4_boundless_state * self.staticGain])
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [self.force]
        self.current_action = self.initial_action.copy()

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
        self.initial_state = np.array([self.theta / self.thetaMax * self.staticGain,
                                       self.dtheta / self.norm_4_boundless_state * self.staticGain,
                                       self.x / self.xMax * self.staticGain,
                                       self.dx / self.norm_4_boundless_state * self.staticGain])
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [self.force]
        self.current_action = self.initial_action.copy()

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

    def saveModel2XML(self, filename='CartPole.xml', filepath='../config/'):
        rootMsg = {
            'name': 'cartpole',
            'author': 'Yefeng YANG',
            'date': '2023.02.18',
            'E-mail': 'yefeng.yang@connect.polyu.hk; 18B904013@stu.hit.edu.cn'
        }
        rl_baseMsg = {
            'state_dim': self.state_dim,
            'state_num': self.state_num,
            'state_step': self.state_step,
            'state_space': self.state_space,
            'state_range': self.state_range,
            'isStateContinuous': self.isStateContinuous,
            'initial_state': self.initial_state,
            'current_state': self.current_state,
            'next_state': self.next_state,
            'action_dim': self.action_dim,
            'action_step': self.action_step,
            'action_range': self.action_range,
            'action_num': self.action_num,
            'action_space': self.action_space,
            'isActionContinuous': self.isActionContinuous,
            'initial_action': self.initial_action,
            'current_action': self.current_action,
            'is_terminal': self.is_terminal
        }
        physicalMsg = {
            'initTheta': self.initTheta,
            'initX': self.initX,
            'theta': self.theta,
            'x': self.x,
            'dtheta': self.dtheta,
            'dx': self.dx,
            'force': self.force,
            'thetaMax': self.thetaMax,
            'xMax': self.xMax,
            'staticGain': self.staticGain,
            'norm_4_boundless_state': self.norm_4_boundless_state,
            'M': self.M,
            'm': self.m,
            'g': self.g,
            'ell': self.ell,
            'kf': self.kf,
            'fMax': self.fm,
            'dt': self.dt,
            'timeMax': self.timeMax,
            'time': self.time,
            'etheta': self.etheta,
            'ex': self.ex
        }
        imageMsg = {
            'width': self.width,
            'height': self.height,
            'name4image': self.name4image,
            'scale': self.scale,
            'xoffset': self.xoffset,
            'cart_x_pixel': self.cart_x_pixel,
            'cart_y_pixel': self.cart_y_pixel,
            'pole_ell_pixel': self.pole_ell_pixel
        }
        xml_cfg().XML_Create(filename=filepath + filename,
                             rootname='Plant',
                             rootmsg=rootMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='RL_Base',
                                 nodemsg=rl_baseMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='Physical',
                                 nodemsg=physicalMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='Image',
                                 nodemsg=imageMsg)
        xml_cfg().XML_Pretty_All(filepath + filename)

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
