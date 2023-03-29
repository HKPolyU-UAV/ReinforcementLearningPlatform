import numpy as np

from common.common_func import *
from environment.envs import *


class Flight_Attitude_Simulator_Continuous(rl_base):
    def __init__(self, initTheta: float, setTheta: float, save_cfg: bool):
        """
        @note:                  initialization
        @param initTheta:       initial theta
        @param setTheta:        target theta=0
        @param save_cfg:        save model description file?
        """
        super(Flight_Attitude_Simulator_Continuous, self).__init__()
        '''physical parameters'''
        self.name = 'Flight_Attitude_Simulator_Continuous2'
        self.initTheta = deg2rad(initTheta)
        self.setTheta = deg2rad(setTheta)
        self.force = 0
        self.f_max = 3.5
        self.f_min = -1.5

        self.minTheta = deg2rad(-60.0)
        self.maxTheta = deg2rad(60.0)

        self.min_omega = deg2rad(-90)
        self.max_omega = deg2rad(90)

        self.min_theta_e = self.setTheta - self.maxTheta
        self.max_theta_e = self.setTheta - self.minTheta

        self.theta = min(max(self.initTheta, self.minTheta), self.maxTheta)
        self.thetaError = self.setTheta - self.theta
        self.dTheta = 0.0

        self.dt = 0.02  # control period
        self.time = 0.0

        self.sum_thetaError = 0.0
        self.timeMax = 15

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
        # 这个状态与控制系统的状态不一样
        self.staticGain = 2
        self.state_dim = 4  # initTheta, Theta, dTheta, error
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 1
        self.action_step = [None]
        self.action_range = [[self.f_min, self.f_max]]
        self.action_num = [math.inf]
        self.action_space = [None]
        self.isActionContinuous = True
        self.initial_action = np.array([0.0])
        self.current_action = self.initial_action.copy()

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
        self.image = np.zeros([self.width, self.height, 3], np.uint8)
        self.image[:, :, 0] = np.ones([self.width, self.height]) * 255
        self.image[:, :, 1] = np.ones([self.width, self.height]) * 255
        self.image[:, :, 2] = np.ones([self.width, self.height]) * 255
        self.name4image = 'Flight attitude simulator'
        self.scale = 250  # cm -> pixel
        self.ybias = 360  # pixel
        self.base_hor_w = 0.4
        self.base_hor_h = 0.02
        self.base_ver_w = 0.02
        self.base_ver_h = 0.8

        self.show = self.image.copy()
        self.save = self.image.copy()

        self.draw_base()
        self.draw_pendulum()
        self.draw_copper()
        self.show_initial_image(isWait=True)
        '''visualization_opencv'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_dTheta = [self.dTheta]
        self.save_error = [self.thetaError]
        self.save_F = [self.initial_action[0]]
        '''data_save'''
        if save_cfg:
            self.saveModel2XML()

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
        pt1 = np.atleast_1d([int(L0 * np.cos(theta1 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta1 + self.theta) * self.scale)])
        pt2 = np.atleast_1d([int(L0 * np.cos(theta2 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta2 + self.theta) * self.scale)])
        pt3 = np.atleast_1d([int(L0 * np.cos(theta3 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta3 + self.theta) * self.scale)])
        pt4 = np.atleast_1d([int(L0 * np.cos(theta4 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta4 + self.theta) * self.scale)])
        cv.fillPoly(img=self.show, pts=np.array([[pt1, pt2, pt3, pt4]]), color=Color().Red)
        # self.show = self.image.copy()

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

        cv.fillPoly(img=self.show, pts=np.array([[pt1, pt2, pt3, pt4]]), color=Color().Black)
        # self.show = self.image.copy()

    def show_initial_image(self, isWait):
        cv.imshow(self.name4image, self.show)
        cv.waitKey(0) if isWait else cv.waitKey(1)

    def show_dynamic_image(self, isWait=False):
        self.draw_pendulum()
        self.draw_copper()
        cv.imshow(self.name4image, self.show)
        cv.waitKey(0) if isWait else cv.waitKey(1)
        self.save = self.show.copy()
        self.show = self.image.copy()

    def state_norm(self) -> np.ndarray:
        """
        状态归一化
        """
        # # initTheta, Theta, dTheta, error
        #
        _initTheta = (2 * self.initTheta - self.maxTheta - self.minTheta) / (self.maxTheta - self.minTheta) * self.staticGain
        _Theta = (2 * self.theta - self.maxTheta - self.minTheta) / (self.maxTheta - self.minTheta) * self.staticGain
        _dTheta = (2 * self.dTheta - self.max_omega - self.min_omega) / (self.max_omega - self.min_omega) * self.staticGain
        _error = (2 * self.thetaError - self.max_theta_e - self.min_theta_e) / (self.max_theta_e - self.min_theta_e) * self.staticGain
        # norm_state = np.array([_Theta, _dTheta, _error])

        norm_state = np.array([self.initTheta, self.theta, self.dTheta, self.thetaError])
        # norm_state = np.array([self.theta, self.dTheta, self.thetaError])

        return norm_state

    def inverse_state_norm(self, s: np.ndarray) -> np.ndarray:
        _initTheta = (s[0] / self.staticGain * (self.maxTheta - self.minTheta) + self.maxTheta + self.minTheta) / 2
        _Theta = (s[1] / self.staticGain * (self.maxTheta - self.minTheta) + self.maxTheta + self.minTheta) / 2
        _dTheta = (s[2] / self.staticGain * (self.max_omega - self.min_omega) + self.max_omega + self.min_omega) / 2
        _error = (s[3] / self.staticGain * (self.max_theta_e - self.min_theta_e) + self.max_theta_e + self.min_theta_e) / 2

        inv_norm_state = np.array([self.initTheta, self.theta, self.dTheta, self.thetaError])

        return inv_norm_state

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
        if self.theta > self.maxTheta + deg2rad(1):
            self.terminal_flag = 1
            # print('超出最大角度')
            return True
        if self.theta < self.minTheta - deg2rad(1):
            self.terminal_flag = 2
            # print('超出最小角度')
            return True
        if self.time > self.timeMax:
            self.terminal_flag = 3
            print('Timeout')
            return True
        if self.is_success():
            self.terminal_flag = 4
            print('Success')
            return True
        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        s = self.inverse_state_norm(self.current_state)
        ss = self.inverse_state_norm(self.next_state)
        current_error = math.fabs(s[3])
        next_error = math.fabs(ss[3])

        new_theta = ss[1]
        new_dtheta = ss[2]

        # if current_error > next_error:  # 如果误差变小
        #     r1 = 3  # + (current_error - next_error) * 20
        # elif current_error < next_error:
        #     r1 = -3  # + (current_error - next_error) * 20
        # else:
        #     if next_error > deg2rad(0.5):  # 如果当前误差大于0.5度
        #         r1 = -3
        #     else:
        #         r1 = 5
        #
        # if self.terminal_flag == 2 or self.terminal_flag == 3:
        #     r2 = -5
        # elif self.terminal_flag == 4:
        #     r2 = 10
        # else:
        #     r2 = 0
        #
        # if new_theta * new_dtheta < 0:
        #     r3 = 5
        # else:
        #     r3 = -5

        r1 = -(next_error ** 2) * 100     # 使用 x'Qx 的形式，试试好不好使
        r2 = -(self.dTheta ** 2) * 1
        r3 = 0

        self.reward = r1 + r2 + r3

    def ode(self, xx: np.ndarray):
        _dtheta = xx[1]
        _ddtheta = (self.force * self.L - self.m * self.g * self.dis - self.k * xx[1]) / (self.J + self.m * self.dis ** 2)
        return np.array([_dtheta, _ddtheta])

    def rk44(self, action: float):
        self.force = action
        h = self.dt / 10  # RK-44 解算步长
        tt = self.time + self.dt
        while self.time < tt:
            xx_old = np.array([self.theta, self.dTheta])
            K1 = h * self.ode(xx_old)
            K2 = h * self.ode(xx_old + K1 / 2)
            K3 = h * self.ode(xx_old + K2 / 2)
            K4 = h * self.ode(xx_old + K3)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            xx_temp = xx_new.copy()
            self.theta = xx_temp[0]
            self.dTheta = xx_temp[1]
            self.time += h

    def step_update(self, action: np.ndarray):
        self.current_action = action.copy()
        self.current_state = self.state_norm()  # 当前状态

        '''rk44'''
        self.rk44(action=action[0])
        # print('角速度',rad2deg(self.dTheta))
        '''rk44'''

        self.is_terminal = self.is_Terminal()
        self.thetaError = self.setTheta - self.theta
        self.next_state = self.state_norm()  # 下一时刻

        self.get_reward()

        # '''出界处理'''
        # if self.theta > self.maxTheta:                  # 如果超出最大角度限制
        #     self.theta = self.maxTheta
        #     self.dTheta = -0.8 * self.dTheta            # 碰边界速度直接反弹
        # if self.theta < self.minTheta:
        #     self.theta = self.minTheta
        #     self.dTheta = -0.8 * self.dTheta
        # '''出界处理'''

        self.saveData()

    def reset(self):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        self.theta = self.initTheta
        self.dTheta = 0.0
        self.time = 0.0
        self.thetaError = self.setTheta - self.theta
        self.sum_thetaError = 0.0
        self.draw_base()
        self.draw_pendulum()
        self.draw_copper()
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.state_norm()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_dTheta = [self.dTheta]
        self.save_error = [self.thetaError]
        self.save_F = [self.initial_action[0]]
        '''data_save'''

    def reset_random(self):
        """
        :brief:
        :return:
        """
        '''physical parameters'''
        self.initTheta = random.uniform(self.minTheta, self.maxTheta)
        # self.setTheta = random.uniform(self.minTheta, self.maxTheta)
        # print('initTheta: ', rad2deg(self.initTheta))
        # self.setTheta = random.uniform(self.minTheta, self.maxTheta)
        self.theta = self.initTheta
        self.dTheta = 0.0
        self.time = 0.0
        self.thetaError = self.setTheta - self.theta
        self.sum_thetaError = 0.0
        self.draw_base()
        self.draw_pendulum()
        self.draw_copper()
        # self.show_initial_image(isWait=True)
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_dTheta = [self.dTheta]
        self.save_error = [self.thetaError]
        self.save_F = [self.initial_action[0]]
        '''data_save'''

    def saveModel2XML(self, filename='Flight_Attitude_Simulator_Continuous2.xml', filepath='../config/'):
        rootMsg = {
            'name': 'Flight_Attitude_Simulator_Continuous2',
            'author': 'Yefeng YANG',
            'date': '2023.03.02',
            'E-mail': 'yefeng.yang@connect.polyu.hk'
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
            'Q': self.Q,
            'R': self.R,
            'is_terminal': self.is_terminal
        }
        physicalMsg = {
            'initTheta': self.initTheta,
            'setTheta': self.setTheta,
            'force': self.force,
            'f_max': self.f_max,
            'f_min': self.f_min,
            'minTheta': self.minTheta,
            'maxTheta': self.maxTheta,
            'min_omega': self.min_omega,
            'max_omega': self.max_omega,
            'min_theta_e': self.min_theta_e,
            'max_theta_e': self.max_theta_e,
            'theta': self.theta,
            'dTheta': self.dTheta,
            'dt': self.dt,
            'Lw': self.Lw,
            'L': self.L,
            'J': self.J,
            'k': self.k,
            'm': self.m,
            'dis': self.dis,
            'copperl': self.copperl,
            'copperw': self.copperw,
            'g': self.g
        }
        imageMsg = {
            'width': self.width,
            'height': self.height,
            'name4image': self.name4image,
            'scale': self.scale,
            'ybias': self.ybias,
            'base_hor_w': self.base_hor_w,
            'base_hor_h': self.base_hor_h,
            'base_ver_w': self.base_ver_w,
            'base_ver_h': self.base_ver_h
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

    def saveData(self, is2file=False, filename='Flight_Attitude_Simulator2.csv', filepath=''):
        self.save_Time.append(self.time)
        self.save_Theta.append(self.theta)
        self.save_dTheta.append(self.dTheta)
        self.save_error.append(self.thetaError)
        self.save_F.append(self.current_action[0])
        if is2file:
            data = pd.DataFrame({
                'time:': self.save_Time,
                'theta': self.save_Theta,
                'dTheta': self.save_dTheta,
                'thetaError': self.save_error,
                'F': self.save_F
            })
            data.to_csv(filepath + filename, index=False, sep=',')
