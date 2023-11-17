import math
import cv2 as cv
import numpy as np
import pandas as pd
from environment.color import Color
from utils.functions import *
from algorithm.rl_base import rl_base


class BallBalancer1D(rl_base):
    def __init__(self,
                 initTheta: float = 0.0,
                 initPos: float = 0.0,
                 initVel: float = 0.0,
                 target: float = 0.0):
        """
        @note:                  通过一个二轴机械臂支撑平衡小球，机械臂第二个轴保持竖直
        @param initTheta:       initial theta
        @param initPos:         initial position
        @param initVel:         initial velocity
        @param save_cfg:        save model description file?
        """
        super(BallBalancer1D, self).__init__()
        '''physical parameters'''
        self.name = 'BallBalancer1D'
        self.initTheta = initTheta
        self.initVel = initVel
        self.initPos = initPos
        self.target = target

        self.omegaMax = np.pi
        self.omegaMin = -np.pi
        self.thetaMax = deg2rad(45.0)
        self.thetaMin = deg2rad(-45.0)
        self.vMin = -3
        self.vMax = 3

        self.theta = np.clip(self.initTheta, self.thetaMin, self.thetaMax)  # 机械臂第一个轴于水平面夹角
        self.pos = self.initPos  # 小球位置，以摆杆中心向左为正
        self.vel = self.initVel  # 小球速度
        self.omega = 0  # 机械臂转轴转速，作为控制输入
        self.error = self.target - self.pos

        self.dt = 0.02  # control period
        self.time = 0.0

        self.timeMax = 8

        self.m = 0.26  # 小球质量
        self.rBall = 0.02  # 小球半径
        self.rMotor = 0.0245  # 机械臂长
        self.L = 0.134  # 摆杆半长
        self.Lw = 0.005  # 杆宽
        self.g = 9.81  # 重力加速度
        self.J = 0.0000416  # 小球转动惯量
        self.K = (self.m * self.g * self.rBall ** 2 * self.rMotor) / (
                (self.m * self.rBall ** 2 + self.J) * self.L)  # 动力学方程系数
        self.alpha = np.arcsin(self.rMotor * np.sin(self.theta) / self.L)  # 摆杆摆角
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.staticGain = 2
        self.state_dim = 3  # error, vel, theta
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.use_norm = True
        if self.use_norm:
            self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        else:
            self.state_range = [[-self.L, self.L],
                                [self.vMin, self.vMax],
                                [self.thetaMin, self.thetaMax]]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.action_dim = 1
        self.action_step = [None]
        self.action_range = np.array([[self.omegaMin, self.omegaMax]])
        self.action_num = [math.inf]
        self.action_space = [None]
        self.isActionContinuous = True
        self.current_action = np.zeros(self.action_dim)

        self.reward = 0.0
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        self.is_terminal = False
        '''RL_BASE'''

        '''visualization_opencv'''
        self.width = 400
        self.height = 400
        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.name4image = 'Ball Balancer 1D'
        self.scale = 800  # cm -> pixel
        self.ybias = 250  # pixel
        self.base_hor_w = 0.05
        self.base_hor_h = 0.005
        self.base_ver_w = 0.005
        self.base_ver_h = 0.1

        self.show = self.image.copy()
        self.save = self.image.copy()
        '''visualization_opencv'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_Pos = [self.pos]
        self.save_Vel = [self.vel]
        self.save_error = [self.error]
        self.save_omega = [self.current_action[0]]
        '''data_save'''

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
        self.show = self.image.copy()

    def draw_pendulum(self):
        """
        :brief:     绘制摆杆
        :return:    None
        """
        cx = int(self.width / 2)
        cy = int(self.ybias - (self.base_hor_h + self.base_ver_h) * self.scale)
        theta1 = np.arctan(self.Lw / self.L / 2)
        theta2 = -theta1
        theta3 = math.pi + theta1
        theta4 = math.pi + theta2
        L0 = np.sqrt((self.Lw / 2) ** 2 + self.L ** 2)
        pt1 = np.atleast_1d([int(L0 * np.cos(theta1 + self.alpha) * self.scale + cx),
                             int(cy - L0 * np.sin(theta1 + self.alpha) * self.scale)])
        pt2 = np.atleast_1d([int(L0 * np.cos(theta2 + self.alpha) * self.scale + cx),
                             int(cy - L0 * np.sin(theta2 + self.alpha) * self.scale)])
        pt3 = np.atleast_1d([int(L0 * np.cos(theta3 + self.alpha) * self.scale + cx),
                             int(cy - L0 * np.sin(theta3 + self.alpha) * self.scale)])
        pt4 = np.atleast_1d([int(L0 * np.cos(theta4 + self.alpha) * self.scale + cx),
                             int(cy - L0 * np.sin(theta4 + self.alpha) * self.scale)])
        cv.fillPoly(img=self.show, pts=np.array([[pt1, pt2, pt3, pt4]]), color=Color().Red)
        # self.show = self.image.copy()

    def draw_ball(self):
        """
        :brief:     绘制小球
        :return:    None
        """
        cx = int(self.width / 2)
        cy = int(self.ybias - (self.base_hor_h + self.base_ver_h) * self.scale)
        posPixel = int(self.pos * self.scale)
        rBallPixel = int(self.rBall * self.scale)
        centerBall = np.atleast_1d([int(cx - posPixel * np.cos(self.alpha) - rBallPixel * np.sin(self.alpha)),
                                    int(cy + posPixel * np.sin(self.alpha) - rBallPixel * np.cos(self.alpha))])
        cv.circle(img=self.show, center=centerBall, radius=rBallPixel, color=Color().Black, thickness=-1)
        # self.show = self.image.copy()

    def draw_arm(self):
        """
        :brief: 绘制机械臂
        @return: None
        """
        cx = int(self.width / 2)
        cy = int(self.ybias - (self.base_hor_h + self.base_ver_h) * self.scale)
        pos1 = np.atleast_1d([int(cx + self.L * self.scale * np.cos(self.alpha)),
                              int(cy - self.L * self.scale * np.sin(self.alpha))])  # 摆杆右端点
        pos2 = np.atleast_1d([pos1[0], pos1[1] + int(self.rMotor * self.scale)])  # 机械臂中间轴位置
        posMotor = np.atleast_1d([int(pos2[0] + self.rMotor * np.cos(self.theta) * self.scale),
                                  int(pos2[1] + self.rMotor * np.sin(self.theta) * self.scale)])
        cv.circle(img=self.show, center=posMotor, radius=5, color=Color().Cyan, thickness=-1)
        cv.line(img=self.show, pt1=pos1, pt2=pos2, color=Color().DarkGray, thickness=2)
        cv.line(img=self.show, pt1=pos2, pt2=posMotor, color=Color().DarkGray, thickness=2)

    def draw_init_image(self):
        self.draw_base()
        self.draw_pendulum()
        self.draw_ball()
        self.draw_arm()
        cv.waitKey(1)

    def visualization(self):
        self.draw_base()
        self.draw_pendulum()
        self.draw_ball()
        self.draw_arm()
        cv.imshow(self.name4image, self.show)
        cv.waitKey(1)
        self.save = self.show.copy()
        self.show = self.image.copy()

    def get_state(self) -> np.ndarray:
        """
        @return:
        """
        if self.use_norm:
            _pos = self.pos / self.L * self.staticGain
            _vel = (2 * self.vel - self.vMax - self.vMin) / (self.vMax - self.vMin) * self.staticGain
            _theta = (2 * self.theta - self.thetaMax - self.thetaMin) / (self.thetaMax - self.thetaMin) * self.staticGain
            state = np.array([_pos, _vel, _theta])
        else:
            state = np.array([self.pos, self.vel, self.theta])
        return state

    def is_success(self):
        if np.fabs(self.error) <= 0.001 and np.fabs(self.vel) <= 0.005 and np.fabs(self.theta) <= deg2rad(1):  # 速度也很小
            return True
        return False

    def is_Terminal(self, param=None):
        """
        :brief:     判断回合是否结束
        :return:    是否结束
        """
        if self.pos < -self.L or self.pos > self.L:
            self.terminal_flag = 1
            # print('out')
            return True
        if self.time > self.timeMax:
            self.terminal_flag = 2
            # print('Timeout')
            return True
        if self.is_success():
            self.terminal_flag = 3
            # print('Success')
            return True
        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        # w = (2 * self.omega - self.omegaMax - self.omegaMin) / (self.omegaMax - self.omegaMin) * self.staticGain
        e = self.error / self.L * self.staticGain
        # 二次型奖励 由于theta与pos有比例关系，所以只对pos和v做要求
        r1 = - e ** 2 - np.tanh(100 * e) + 0.5
        # r2 = - 1 * self.next_state[1] ** 2
        r2 = 0
        r3 = 1000 if self.is_success() else 0
        self.reward = r1 + r2 + r3

    def ode(self, xx: np.ndarray):
        _dPos = xx[1]
        _dVel = self.K * np.sin(xx[2])
        _dTheta = self.omega
        return np.array([_dPos, _dVel, _dTheta])

    def rk44(self, action: float):
        self.omega = np.clip(action, self.omegaMin, self.omegaMax)
        h = self.dt / 10  # RK-44 解算步长
        tt = self.time + self.dt
        while self.time < tt:
            xx_old = np.array([self.pos, self.vel, self.theta])
            K1 = h * self.ode(xx_old)
            K2 = h * self.ode(xx_old + K1 / 2)
            K3 = h * self.ode(xx_old + K2 / 2)
            K4 = h * self.ode(xx_old + K3)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            xx_temp = xx_new.copy()
            self.pos = xx_temp[0]
            self.vel = np.clip(xx_temp[1], self.vMin, self.vMax)
            self.theta = np.clip(xx_temp[2], self.thetaMin, self.thetaMax)
            self.alpha = np.arcsin(self.rMotor * np.sin(self.theta) / self.L)
            self.time += h

    def step_update(self, action: np.ndarray):
        self.current_action = action.copy()
        self.current_state = self.get_state()

        '''rk44'''
        self.rk44(action=action[0])
        '''rk44'''

        self.is_terminal = self.is_Terminal()
        self.error = self.target - self.pos
        self.next_state = self.get_state()

        self.get_reward()

        self.saveData()

    def reset(self, random: bool = False):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        if random:
            self.initTheta = np.random.uniform(deg2rad(-40), deg2rad(40))
            self.initPos = np.random.uniform(-0.12, 0.12)  # 防止一开始就掉下去
        self.theta = self.initTheta
        self.pos = self.initPos
        self.vel = self.initVel
        self.omega = 0.0
        self.time = 0.0
        self.error = self.target - self.pos
        self.alpha = np.arcsin(self.rMotor * np.sin(self.theta) / self.L)
        self.draw_init_image()
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()
        self.current_action = np.zeros(self.action_dim)
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_Pos = [self.pos]
        self.save_Vel = [self.vel]
        self.save_error = [self.error]
        self.save_omega = [self.current_action[0]]
        '''data_save'''

    def saveData(self, is2file=False, filename='Flight_Attitude_Simulator_2State_Continuous.csv', filepath=''):
        self.save_Time.append(self.time)
        self.save_Theta.append(self.theta)
        self.save_Pos.append(self.pos)
        self.save_Vel.append(self.vel)
        self.save_error.append(self.error)
        self.save_omega.append(self.current_action[0])
        if is2file:
            data = pd.DataFrame({
                'time:': self.save_Time,
                'theta': self.save_Theta,
                'pos': self.save_Pos,
                'vel': self.save_Vel,
                'error': self.save_error,
                'omega': self.save_omega
            })
            data.to_csv(filepath + filename, index=False, sep=',')
