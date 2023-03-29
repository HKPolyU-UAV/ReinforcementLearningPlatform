import numpy as np

from common.common_func import *
# from common.common_cls import *
from environment.envs import *
from environment.envs.pathplanning.samplingmap import samplingmap


class UGV_Bidirectional_Continuous(samplingmap, rl_base):
    def __init__(self, initPhi: float, save_cfg: bool, x_size: float, y_size: float, start: list, terminal: list):
        """
        :param initPhi:     initial phi
        :param save_cfg:    svae to model file or not
        """
        super(UGV_Bidirectional_Continuous, self).__init__(width=400,
                                                           height=400,
                                                           x_size=x_size,
                                                           y_size=y_size,
                                                           image_name='TwoWheelUGV',
                                                           start=start,
                                                           terminal=terminal,
                                                           obs=None,
                                                           draw=False)
        '''physical parameters'''
        self.initX = self.start[0]
        self.initY = self.start[1]
        self.initPhi = initPhi
        self.x = self.initX  # X
        self.y = self.initY  # Y
        self.phi = self.initPhi  # 车的转角
        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.

        self.wMax = 10  # 车轮最大角速度rad/s
        self.r = 0.1  # 车轮半径
        self.l_wheel = 0.06  # 车轮厚度
        self.rBody = 0.15  # 车主体半径
        self.L = 2 * self.rBody  # 车主体直径
        self.dt = 0.02  # 50Hz
        self.time = 0.  # time
        self.timeMax = 8.0
        self.vMax = 5   # 最大速度 5 m/s
        self.vMin = -5
        self.omegaMax = deg2rad(360)    # 最大角速度 2pi
        self.omegaMin = deg2rad(-360)

        self.miss = 0.4
        self.name = 'UGVBidirectional'
        '''physical parameters'''

        '''rl_base'''
        self.static_gain = 2
        self.state_dim = 8  # ex, ey, x, y, phi, dx, dy, dphi
        self.state_num = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None, None, None, None, None, None, None, None]
        self.state_space = [None, None, None, None, None, None, None, None]
        self.state_range = [[-self.x_size, self.x_size],
                            [-self.y_size, self.y_size],
                            [0, self.x_size],
                            [0, self.y_size],
                            [-math.inf, math.inf],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [self.r / self.L * 2 * self.r * self.wMax, -self.r / self.L * 2 * self.r * self.wMax]]
        self.isStateContinuous = [True, True, True, True, True, True, True, True]
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 2
        self.action_step = [None, None]
        self.action_range = [[-self.wMax, self.wMax], [-self.wMax, self.wMax]]
        self.action_num = [math.inf, math.inf]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.initial_action = [0.0, 0.0]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        '''visualization_opencv'''
        self.show_dynamic_image(isWait=False)
        '''visualization_opencv'''

        '''datasave'''
        self.saveX = [self.x]
        self.saveY = [self.y]
        self.savePhi = [self.phi]
        self.savedX = [self.dx]
        self.savedY = [self.dy]
        self.savedPhi = [self.dphi]
        self.savewLeft = [self.wLeft]
        self.savewRight = [self.wRight]
        self.saveTime = [self.time]
        '''datasave'''
        if save_cfg:
            self.saveModel2XML()

    def draw_car(self):
        self.image = self.image_temp.copy()
        rBody = self.rBody * 3
        r = self.r * 2
        l_wheel = self.l_wheel * 2
        cv.circle(self.image, self.dis2pixel([self.x, self.y]), self.length2pixel(rBody), Color().Orange, -1)  # 主体
        '''两个车轮'''
        # left
        pts_left = [[r, rBody], [r, rBody - l_wheel], [-r, rBody - l_wheel], [-r, rBody]]
        pts_left = points_rotate(pts_left, self.phi)
        pts_left = points_move(pts_left, [self.x, self.y])
        cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in pts_left]), color=Color().Red)
        # right
        pts_right = [[r, -rBody], [r, -rBody + l_wheel], [-r, -rBody + l_wheel], [-r, -rBody]]
        pts_right = points_rotate(pts_right, self.phi)
        pts_right = points_move(pts_right, [self.x, self.y])
        cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in pts_right]), color=Color().Red)
        '''两个车轮'''
        # 额外画一个圆形，标志头
        head = [r - 0.02, 0]
        head = points_rotate(head, self.phi)
        head = points_move(head, [self.x, self.y])
        cv.circle(self.image, self.dis2pixel(head), self.length2pixel(0.1), Color().Black, -1)  # 主体

    def draw_terminal(self):
        if self.terminal is not None and self.terminal != []:
            cv.circle(self.image, self.dis2pixel(self.terminal), 5, Color().random_color_by_BGR(), -1)

    def draw_region_grid(self, xNUm: int = 3, yNum: int = 3):
        if xNUm <= 1 or yNum <= 1:
            pass
        else:
            xStep = self.x_size / xNUm
            yStep = self.y_size / yNum
            for i in range(yNum - 1):
                pt1 = self.dis2pixel([0, 0 + (i + 1) * yStep])
                pt2 = self.dis2pixel([self.x_size, 0 + (i + 1) * yStep])
                cv.line(self.image, pt1, pt2, Color().Black, 1)
            for i in range(xNUm - 1):
                pt1 = self.dis2pixel([0 + (i + 1) * xStep, 0])
                pt2 = self.dis2pixel([0 + (i + 1) * xStep, self.y_size])
                cv.line(self.image, pt1, pt2, Color().Black, 1)

    def show_dynamic_image(self, isWait):
        self.image_temp = self.image.copy()
        self.map_draw_boundary()
        self.draw_car()
        self.draw_terminal()
        self.draw_region_grid(xNUm=3, yNum=3)

        cv.putText(self.image, str(round(self.time, 3)), (0, 15), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 1)
        cv.imshow(self.name4image, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)
        self.save = self.image.copy()
        self.image = self.image_temp.copy()

    def state_norm(self):
        # [self.terminal[0] - self.x, self.terminal[1] - self.y, self.x, self.y, self.phi, self.dx, self.dy, self.dphi]
        _ex = (self.terminal[0] - self.x) / self.x_size * self.static_gain
        _ey = (self.terminal[1] - self.y) / self.y_size * self.static_gain
        _x = (2 * self.x / self.x_size - 1) * self.static_gain
        _y = (2 * self.y / self.y_size - 1) * self.static_gain
        _phi = self.phi / np.pi * self.static_gain
        _dx = self.dx / self.vMax * self.static_gain
        _dy = self.dy / self.vMax * self.static_gain
        _dphi = self.dphi / self.omegaMax * self.static_gain
        return np.array([_ex, _ey, _x, _y, _phi, _dx, _dy, _dphi])

    def inverse_state_norm(self, s: np.ndarray):
        # [self.terminal[0] - self.x, self.terminal[1] - self.y, self.x, self.y, self.phi, self.dx, self.dy, self.dphi]
        s[0] = s[0] / self.static_gain * self.x_size
        s[1] = s[1] / self.static_gain * self.y_size
        s[2] = (s[2] / self.static_gain + 1) * self.x_size / 2
        s[3] = (s[3] / self.static_gain + 1) * self.y_size / 2
        s[4] = s[4] / self.static_gain * np.pi
        s[5] = s[5] / self.static_gain * self.vMax
        s[6] = s[6] / self.static_gain * self.vMax
        s[7] = s[7] / self.static_gain * self.omegaMax
        return s

    def is_out(self):
        """
        :return:
        """
        '''简化处理，只判断中心的大圆有没有出界就好'''
        if (self.x + self.rBody > self.x_size) or (self.x - self.rBody < 0) or (self.y + self.rBody > self.y_size) or (self.y - self.rBody < 0):
            return True
        return False

    def is_success(self):
        ex = self.terminal[0] - self.x
        ey = self.terminal[1] - self.y
        if np.linalg.norm([ex, ey]) <= self.miss and np.fabs(self.dphi) < deg2rad(1):
            return True
        return False

    def is_Terminal(self, param=None):
        # if self.is_out():
        #     print('...out...')
        #     self.terminal_flag = 1
        #     return True
        # if math.fabs(self.initPhi - self.phi) > 2 * math.pi:
        #     print('...转的角度太大了...')
        #     self.terminal_flag = 1
        #     return True
        if self.time > self.timeMax:
            print('...time out...')
            self.terminal_flag = 2
            return True
        # if dis_two_points([self.x, self.y], self.terminal) <= self.miss:
        if self.is_success():
            print('...success...')
            self.terminal_flag = 3
            return True
        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        cur_s = self.inverse_state_norm(self.current_state)
        nex_s = self.inverse_state_norm(self.next_state)

        currentError = math.sqrt(cur_s[0] ** 2 + cur_s[1] ** 2)
        nextError = math.sqrt(nex_s[0] ** 2 + nex_s[1] ** 2)

        r1 = -1  # 常值误差，每运行一步，就 -1

        # if currentError > nextError + 1e-2:
        #     r2 = 3
        # elif 1e-2 + currentError < nextError:
        #     r2 = -3
        # else:
        #     r2 = 0
        r2 = 0
        # currentTheta = cal_vector_rad([cur_s[0], cur_s[1]], [math.cos(cur_s[4]), math.sin(cur_s[4])])
        # nextTheta = cal_vector_rad([nex_s[0], nex_s[1]], [math.cos(nex_s[4]), math.sin(nex_s[4])])
        # # print(currentTheta, nextTheta)
        # if currentTheta > nextTheta + 1e-3:  # 带1e-4是为了
        #     r3 = 2
        # elif 1e-3 + currentTheta < nextTheta:
        #     r3 = -2
        # else:
        #     r3 = 0
        r3 = 0
        '''4. 其他'''
        if self.terminal_flag == 3:     # 成功
            r4 = 1000
        elif self.terminal_flag == 2:   # 超时
            r4 = -200
        else:
            r4 = 0
        # if self.terminal_flag == 1:  # 惩
        #     r4 = -200
        '''4. 其他'''
        # print('r1=', r1, 'r2=', r2, 'r3=', r3, 'r4=', r4)
        self.reward = r1 + r2 + r3 + r4

    def f(self, _phi):
        _dx = self.r / 2 * (self.wLeft + self.wRight) * math.cos(_phi)
        _dy = self.r / 2 * (self.wLeft + self.wRight) * math.sin(_phi)
        _dphi = self.r / self.rBody * (self.wRight - self.wLeft)
        return np.array([_dx, _dy, _dphi])

    def step_update(self, action: list):
        self.wLeft = max(min(action[0], self.wMax), -self.wMax)
        self.wRight = max(min(action[1], self.wMax), -self.wMax)
        self.current_action = action.copy()
        self.current_state = self.state_norm()

        '''RK-44'''
        h = self.dt / 10
        t_sim = 0
        state = np.array([self.x, self.y, self.phi])
        while t_sim <= self.dt:
            K1 = self.f(state[2])
            K2 = self.f(state[2] + h * K1[2] / 2)
            K3 = self.f(state[2] + h * K2[2] / 2)
            K4 = self.f(state[2] + h * K3[2])
            state = state + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6
            t_sim += h
        '''动力学系统状态更新'''
        [self.x, self.y, self.phi] = list(state)
        self.dx = self.r / 2 * (self.wLeft + self.wRight) * math.cos(self.phi)
        self.dy = self.r / 2 * (self.wLeft + self.wRight) * math.sin(self.phi)
        self.dphi = self.r / self.rBody * (self.wRight - self.wLeft)
        self.time += self.dt
        '''动力学系统状态更新'''
        '''RK-44'''

        '''出界处理'''
        # if self.x + self.rBody > self.x_size:  # Xout
        #     self.x = self.x_size - self.rBody
        #     self.dx = 0
        # if self.x - self.rBody < 0:
        #     self.x = self.rBody
        #     self.dx = 0
        # if self.y + self.rBody > self.y_size:  # Yout
        #     self.y = self.y_size - self.rBody
        #     self.dy = 0
        # if self.y - self.rBody < 0:
        #     self.y = self.rBody
        #     self.dy = 0
        '''出界处理'''

        if self.phi > np.pi:
            self.phi -= 2 * np.pi
        if self.phi < -np.pi:
            self.phi += 2 * np.pi
        self.is_terminal = self.is_Terminal()
        self.next_state = self.state_norm()
        self.get_reward()
        self.saveData()

    def reset(self):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        self.x = self.initX  # X
        self.y = self.initY  # Y
        self.phi = self.initPhi  # 车的转角
        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.
        self.time = 0.  # time
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.saveX = [self.x]
        self.saveY = [self.y]
        self.savePhi = [self.phi]
        self.savedX = [self.dx]
        self.savedY = [self.dy]
        self.savedPhi = [self.dphi]
        self.savewLeft = [self.wLeft]
        self.savewRight = [self.wRight]
        '''data_save'''

    def reset_random(self):
        """
        :return:
        """
        '''physical parameters'''
        self.set_terminal([random.uniform(self.L, self.x_size - self.L), random.uniform(self.L, self.y_size - self.L)])
        self.set_start([random.uniform(self.L, self.x_size - self.L), random.uniform(self.L, self.y_size - self.L)])
        # self.x = self.initX  # X
        # self.y = self.initY  # Y
        self.x = self.start[0]
        self.y = self.start[1]
        self.phi = self.initPhi  # 车的转角
        # self.phi = random.uniform(deg2rad(-180), deg2rad(180))
        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.
        self.time = 0.  # time
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.state_norm()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.saveX = [self.x]
        self.saveY = [self.y]
        self.savePhi = [self.phi]
        self.savedX = [self.dx]
        self.savedY = [self.dy]
        self.savedPhi = [self.dphi]
        self.savewLeft = [self.wLeft]
        self.savewRight = [self.wRight]
        '''data_save'''

    def saveModel2XML(self, filename='UGV_Bidirectional_Continuous.xml', filepath='../config/'):
        rootMsg = {
            'name': 'UGV_Bidirectional_Continuous',
            'author': 'Yefeng YANG',
            'date': '2022.01.11',
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
            'initX': self.initX,
            'initY': self.initY,
            'initPhi': self.initPhi,
            'x': self.x,
            'y': self.y,
            'phi': self.phi,
            'dx': self.dx,
            'dy': self.dy,
            'dphi': self.dphi,
            'wLeft': self.wLeft,
            'wRight': self.wRight,
            'wMax': self.wMax,
            'r': self.r,
            'l_wheel': self.l_wheel,
            'rBody': self.rBody,
            'L': self.L,
            'dt': self.dt,
            'time': self.time,
            'x_size': self.x_size,
            'y_size': self.y_size,
            'miss': self.miss
        }
        imageMsg = {
            'width': self.width,
            'height': self.height,
            'name4image': self.name4image,
            'x_offset': self.x_offset,
            'y_offset': self.y_offset,
            'pixel_per_meter': self.pixel_per_meter
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

    def saveData(self, is2file=False, filename='Two_Wheel_UGV.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'x:': self.saveX,
                'y': self.saveY,
                'phi': self.savePhi,
                'dx': self.savedX,
                'dy': self.savedY,
                'dphi': self.savedPhi,
                'wLeft': self.savewLeft,
                'wRight': self.savewRight,
                'time': self.saveTime
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.saveX = [self.x]
            self.saveY = [self.y]
            self.savePhi = [self.phi]
            self.savedX = [self.dx]
            self.savedY = [self.dy]
            self.savedPhi = [self.dphi]
            self.savewLeft = [self.wLeft]
            self.savewRight = [self.wRight]
            self.saveTime = [self.time]
