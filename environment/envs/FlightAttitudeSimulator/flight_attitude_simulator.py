from common.common_func import *
from environment.envs import *


class Flight_Attitude_Simulator(rl_base):
    def __init__(self, initTheta: float, setTheta: float, save_cfg: bool):
        """
        :brief:                 initialization
        :param initTheta:       initial theta
        :param setTheta:        set Theta
        """
        super(Flight_Attitude_Simulator, self).__init__()
        '''physical parameters'''
        self.initTheta = deg2rad(initTheta)
        self.setTheta = deg2rad(setTheta)
        self.f_max = 3.0
        self.f_min = -1.6
        self.f_step = 0.1
        self.minTheta = deg2rad(-60.0)
        self.maxTheta = deg2rad(60.0)
        self.theta = max(self.initTheta, self.minTheta)
        self.theta = min(self.initTheta, self.maxTheta)
        self.dTheta = 0.0
        self.freq = 100  # control frequency
        self.T = 1 / self.freq  # control period
        self.time = 0.0
        self.thetaError = self.setTheta - self.theta
        self.sum_thetaError = 0.0
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.state_dim = 4  # initTheta, Theta, dTheta, error
        self.state_num = [math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[self.minTheta, self.maxTheta],
                            [self.minTheta, self.maxTheta],
                            [-math.inf, math.inf],
                            [self.minTheta - self.maxTheta, self.maxTheta - self.minTheta]]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.initial_state = [self.initTheta, self.theta, self.dTheta, self.thetaError]
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

        self.Lw = 0.02  # 杆宽度
        self.L = 0.362  # 杆半长
        self.J = 0.082  # 转动惯量
        self.k = 0.09  # 摩擦系数
        self.m = 0.3  # 配重重量
        self.dis = 0.059  # 铜块中心距中心距离0.059
        self.copperl = 0.06  # 铜块长度
        self.copperw = 0.03  # 铜块宽度
        self.g = 9.8  # 重力加速度

        self.show = self.image.copy()
        self.save = self.image.copy()

        self.terminal_flag = 0  # 0-正常 1-上边界出界 2-下边界出界 3-超时

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

    def is_Terminal(self, param=None):
        """
        :brief:     判断回合是否结束
        :return:    是否结束
        """
        # if self.theta > self.maxTheta + deg2rad(1):
        #     self.terminal_flag = 1
        #     print('超出最大角度')
        #     return True
        # if self.theta < self.minTheta - deg2rad(1):
        #     self.terminal_flag = 2
        #     print('超出最小角度')
        #     return True
        if self.time > 2:
            self.terminal_flag = 3
            print('超时')
            return True
        self.terminal_flag = 0
        return False

    def step_update(self, action):
        _action = action[0]

        def f(angle, dangle):
            a2 = -self.k / (self.J + self.m * self.dis ** 2)
            a1 = -self.m * self.g * self.dis / (self.dis + self.m * self.dis ** 2)
            a0 = self.L * _action / (self.J + self.m * self.dis ** 2)
            return a2 * dangle + a1 * np.cos(angle) + a0

        h = self.T / 10
        t_sim = 0.0
        self.current_state = [self.initTheta, self.theta, self.dTheta, self.thetaError]
        '''differential equation'''
        while t_sim <= self.T:
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
        if self.theta > self.maxTheta:
            self.theta = self.maxTheta
            self.dTheta = -0.8 * self.dTheta
        if self.theta < self.minTheta:
            self.theta = self.minTheta
            self.dTheta = -0.8 * self.dTheta
        self.time = self.time + self.T
        self.thetaError = self.setTheta - self.theta
        self.sum_thetaError = self.sum_thetaError + abs(self.thetaError)
        self.next_state = [self.initTheta, self.theta, self.dTheta, self.thetaError]
        '''differential equation'''

        '''is_terminal'''
        self.is_terminal = self.is_Terminal()
        '''is_terminal'''

        '''reward function'''
        '''1. 角度误差'''
        gain = 20.0
        r1 = -gain * self.thetaError ** 2
        '''1. 角度误差'''

        '''2. 角速度误差'''
        r2 = 0
        '''2. 角速度误差'''

        '''3. 累计角度误差'''
        r3 = 0
        '''3. 累计角度误差'''

        '''4. 其他误差'''
        r4 = 0
        # if (self.terminal_flag == 1) or (self.terminal_flag == 2):
        #     r4 = -500 * (self.maxTheta - self.minTheta) ** 2
        '''4. 其他误差'''

        self.reward = r1 + r2 + r3 + r4
        '''reward function'''

        self.saveData()

        return self.current_state, action, self.reward, self.next_state, self.is_terminal

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
        '''physical parameters'''

        '''RL_BASE'''
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

    def reset_random(self):
        """
        :brief:
        :return:
        """
        '''physical parameters'''
        # while True:     # 刻意让初始化范围固定在大于0
        #     self.initTheta = random.uniform(self.minTheta, self.maxTheta)
        #     if self.initTheta >= 0.:
        #         break
        self.initTheta = random.uniform(self.minTheta, self.maxTheta)
        print('initTheta: ', rad2deg(self.initTheta))
        # self.setTheta = random.uniform(self.minTheta, self.maxTheta)
        self.theta = self.initTheta
        self.dTheta = 0.0
        self.time = 0.0
        self.thetaError = self.setTheta - self.theta
        self.sum_thetaError = 0.0
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.initial_state = [self.initTheta, self.theta, self.dTheta, self.thetaError]
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

    def saveModel2XML(self, filename='Flight_Attitude_Simulator.xml', filepath='../config/'):
        rootMsg = {
            'name': 'Flight_Attitude_Simulator',
            'author': 'Yefeng YANG',
            'date': '2021.12.27',
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
            'setTheta': self.setTheta,
            'f_max': self.f_max,
            'f_min': self.f_min,
            'f_step': self.f_step,
            'minTheta': self.minTheta,
            'maxTheta': self.maxTheta,
            'theta': self.theta,
            'dTheta': self.dTheta,
            'freq': self.freq,
            'T': self.T,
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

    def saveData(self, is2file=False, filename='Flight_Attitude_Simulator.csv', filepath=''):
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
