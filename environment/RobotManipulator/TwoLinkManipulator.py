import cv2 as cv
from environment.color import Color
from utils.functions import *
from algorithm.rl_base import rl_base


class TwoLinkManipulator(rl_base):
    def __init__(self,
                 theta0: np.ndarray = np.array([0.0, 0.0]),
                 omega0: np.ndarray = np.array([0.0, 0.0]),
                 map_size: np.ndarray = np.array([2.0, 2.0]),
                 target: np.ndarray = np.array([0.5, 0.5])):
        super(TwoLinkManipulator, self).__init__()

        self.init_theta = theta0
        self.init_omega = omega0
        self.init_target = target
        self.init_midPos = np.array([1.0, 0.65])
        self.init_endPos = np.array([1.0, 0.3])

        self.theta = theta0  # 两杆的角度，分别是杆1与y轴负半轴夹角、杆2与杆1延长线夹角，逆时针为正
        self.omega = omega0  # 两杆的角速度
        self.torque = np.zeros(2, dtype=np.float32)  # 两个转轴输入转矩
        self.map_size = map_size
        self.target = target
        self.midPos = self.init_midPos.copy()  # 两杆铰接位置
        self.endPos = self.init_endPos.copy()  # 末端位置
        self.endVel = np.zeros(2)
        self.error = self.target - self.endPos

        '''hyper-parameters'''
        self.basePos = np.array([1.0, 1.0])  # 基座位置
        self.l = 0.35  # 杆长
        self.m = 0.5  # 单根杆质量
        self.g = 9.8  # 重力加速度
        self.J = self.m * (self.l ** 2) / 3  # 杆的转动惯量
        self.dt = 0.02  # 50Hz
        self.time = 0.  # time
        self.time_max = 8.0
        '''hyper-parameters'''

        self.thetaMax = np.pi
        self.omegaMax = np.pi
        self.eMax = 4 * self.l
        self.torqueMax = 5.0

        self.miss = 0.01  # 容许误差
        self.name = 'TwoLinkManipulator'

        '''rl_base'''
        self.static_gain = 2
        self.state_dim = 6  # error theta omega 末端误差 角度 角速度
        self.state_num = [np.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.use_norm = True
        if self.use_norm:
            self.state_range = np.array([[-self.static_gain, self.static_gain],
                                         [-self.static_gain, self.static_gain],
                                         [-self.static_gain, self.static_gain],
                                         [-self.static_gain, self.static_gain],
                                         [-self.static_gain, self.static_gain],
                                         [-self.static_gain, self.static_gain]])
        else:
            self.state_range = np.array(
                [[-self.eMax, self.eMax],
                 [-self.eMax, self.eMax],
                 [-self.thetaMax, self.thetaMax],
                 [-self.thetaMax, self.thetaMax],
                 [-self.omegaMax, self.omegaMax],
                 [-self.omegaMax, self.omegaMax]]
            )
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.action_dim = 2  # 两个转轴转矩
        self.action_step = [None, None]
        self.action_range = np.array(
            [[-self.torqueMax, self.torqueMax],
             [-self.torqueMax, self.torqueMax]])
        self.action_num = [np.inf, np.inf]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.current_action = np.zeros(self.action_dim)

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        '''visualization'''
        self.x_offset = 20
        self.y_offset = 20
        self.board = 250
        self.pixel_per_meter = 100
        self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array(
            [self.x_offset, self.y_offset])).astype(int)
        self.image_size[0] += self.board
        self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
        self.image_white = self.image.copy()  # 纯白图
        self.base_x_pixel = 25
        self.base_y_pixel = 10
        '''visualization'''

        self.sum_d_theta = np.zeros(2)

    def dis2pixel(self, coord) -> tuple:
        x = self.x_offset + coord[0] * self.pixel_per_meter
        y = self.image_size[1] - self.y_offset - coord[1] * self.pixel_per_meter
        return int(x), int(y)

    def length2pixel(self, _l):
        return int(_l * self.pixel_per_meter)

    def draw_text(self):
        cv.putText(
            self.image,
            'time:  %.2fs' % (self.time),
            (self.image_size[0] - self.board - 5, 25), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

        cv.putText(
            self.image,
            'error: [%.2f, %.2f]m' % (self.error[0], self.error[1]),
            (self.image_size[0] - self.board - 5, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'theta: [%.1f, %.1f]' % (rad2deg(self.theta[0]), rad2deg(self.theta[1])),
            (self.image_size[0] - self.board - 5, 95), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'omega: [%.1f, %.1f]' % (rad2deg(self.omega[0]), rad2deg(self.omega[1])),
            (self.image_size[0] - self.board - 5, 130), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

    def draw_init_image(self):
        self.draw_boundary()
        self.image_white = self.image.copy()

    def visualization(self):
        self.image = self.image_white.copy()
        self.draw_boundary()
        self.draw_manipulator()
        self.draw_target()
        self.draw_text()
        cv.imshow(self.name, self.image)
        cv.waitKey(1)

    def draw_manipulator(self):
        # 基座
        base = self.dis2pixel(self.basePos)
        pt1 = (int(base[0] - self.base_x_pixel / 2), int(base[1] - self.base_y_pixel / 2))
        pt2 = (int(base[0] + self.base_x_pixel / 2), int(base[1] + self.base_y_pixel / 2))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().DarkGray, thickness=-1)
        # 杆
        mid = self.dis2pixel(self.midPos)
        end = self.dis2pixel(self.endPos)
        pt1 = (base[0], base[1])
        pt2 = (mid[0], mid[1])
        cv.line(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=4)
        pt1 = (end[0], end[1])
        pt2 = (mid[0], mid[1])
        cv.line(self.image, pt1=pt1, pt2=pt2, color=Color().Red, thickness=4)
        # 末端机械手(迫真)
        cv.line(self.image, pt1=(end[0] - 10, end[1]), pt2=(end[0] + 10, end[1]), color=Color().Chocolate2, thickness=2)
        cv.line(self.image, pt1=(end[0], end[1] - 10), pt2=(end[0], end[1] + 10), color=Color().Chocolate2, thickness=2)

    def draw_boundary(self):
        cv.line(self.image, (self.x_offset, self.y_offset),
                (self.image_size[0] - self.x_offset - self.board, self.y_offset), Color().Black, 2)
        cv.line(self.image, (self.x_offset, self.y_offset), (self.x_offset, self.image_size[1] - self.y_offset),
                Color().Black, 2)
        cv.line(
            self.image,
            (self.image_size[0] - self.x_offset - self.board, self.image_size[1] - self.y_offset),
            (self.x_offset, self.image_size[1] - self.y_offset), Color().Black, 2
        )
        cv.line(
            self.image,
            (self.image_size[0] - self.x_offset - self.board, self.image_size[1] - self.y_offset),
            (self.image_size[0] - self.x_offset - self.board, self.y_offset), Color().Black, 2
        )

    def draw_target(self):
        cv.circle(self.image, self.dis2pixel(self.target), 5, Color().random_color_by_BGR(), -1)

    def get_state(self):
        state = np.concatenate((self.error, self.theta, self.omega), axis=0)
        if self.use_norm:
            norm_min = self.state_range[:, 0]
            norm_max = self.state_range[:, 1]
            state = (2 * state - (norm_min + norm_max)) / (norm_max - norm_min) * self.static_gain
        return state

    def is_success(self):
        b1 = np.linalg.norm(self.error) <= self.miss
        b2 = np.linalg.norm(self.omega) <= deg2rad(5)
        return b1 and b2

    def is_Terminal(self, param=None):
        if self.time > self.time_max:
            # print('...time out...')
            self.terminal_flag = 2
            return True
        if self.is_success():
            print('...success...')
            self.terminal_flag = 3
            return True
        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        Q_pos = 2.0
        Q_omage = 0.1
        Q_acc = 0.005

        r_pos = -np.linalg.norm(self.error) * Q_pos
        r_omega = -np.linalg.norm(self.omega) * Q_omage
        r_acc = -np.linalg.norm(self.torque) * Q_acc

        r_psi = 0.
        # if self.terminal_flag == 4:  # 瞎几把转
        #     _n = (self.time_max - self.time) / self.dt
        #     r_psi = _n * (r_pos + r_omega + r_acc)
        self.reward = r_pos + r_omega + r_acc + r_psi

    def ode(self, xx: np.ndarray):
        [_theta1, _theta2, _omega1, _omega2] = xx[:]
        _dtheta1 = _omega1
        _dtheta2 = _omega2
        a = np.array([[self.J * (5 + 3 * np.cos(_theta2)), self.J * (1 + 3 / 2 * np.cos(_theta2))],
                      [self.J * (1 + 3 / 2 * np.cos(_theta2)), self.J]])
        b = np.array([self.torque[0] + 3 / 2 * self.J * np.sin(_theta2) * (_dtheta2 ** 2) + 3 * self.J * np.sin(
            _theta2) * _dtheta1 * _dtheta2 - self.m * self.g * self.l * (
                              3 / 2 * np.sin(_theta1) + 1 / 2 * np.sin(_theta1 + _theta2)),
                      self.torque[1] - 3 / 2 * self.J * np.sin(_theta2) * (
                              _dtheta1 ** 2) - 1 / 2 * self.m * self.g * self.l * np.sin(_theta1 + _theta2)])
        _domega1, _domega2 = np.linalg.solve(a, b)
        return np.array([_dtheta1, _dtheta2, _domega1, _domega2])

    def rk44(self, action: np.ndarray):
        self.torque = action
        theta = self.theta.copy()
        xx = np.concatenate((self.theta, self.omega))
        K1 = self.dt * self.ode(xx)
        K2 = self.dt * self.ode(xx + K1 / 2)
        K3 = self.dt * self.ode(xx + K2 / 2)
        K4 = self.dt * self.ode(xx + K3)
        xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        self.theta[:] = xx[0: 2]
        self.omega[:] = xx[2: 4]
        self.time += self.dt
        '''正运动学'''
        self.midPos = np.array([self.l * np.sin(self.theta[0]), -self.l * np.cos(self.theta[0])]) + self.basePos
        self.endPos = self.midPos + np.array([self.l * np.sin(sum(self.theta)), -self.l * np.cos(sum(self.theta))])
        self.endVel = np.array([self.l * np.cos(self.theta[0]) * self.omega[0] + self.l * np.cos(sum(self.theta)) * sum(self.omega),
                                self.l * np.sin(self.theta[0]) * self.omega[0] + self.l * np.sin(sum(self.theta)) * sum(self.omega)])
        '''正运动学'''
        self.error = self.target - self.endPos
        self.sum_d_theta += np.fabs(theta - self.theta)
        if self.theta[0] > self.thetaMax:
            self.theta[0] -= 2 * self.thetaMax
        elif self.theta[0] < -self.thetaMax:
            self.theta[0] += 2 * self.thetaMax
        else:
            pass

        if self.theta[1] > self.thetaMax:
            self.theta[1] -= 2 * self.thetaMax
        elif self.theta[1] < -self.thetaMax:
            self.theta[1] += 2 * self.thetaMax
        else:
            pass

    def step_update(self, action: np.ndarray):
        self.current_action = action.copy()
        self.current_state = self.get_state()
        self.rk44(action=action)
        self.is_terminal = self.is_Terminal()
        self.next_state = self.get_state()
        self.get_reward()

    def reset(self, random: bool = True):
        if random:
            '''通过极坐标方式在机械臂动作空间圆内生成随机目标点'''
            phi = np.random.random() * 2 * np.pi
            r = np.random.uniform(0.3 ** 2, (2 * self.l) ** 2)
            x = np.cos(phi) * (r ** 0.5) + self.basePos[0]
            y = np.sin(phi) * (r ** 0.5) + self.basePos[1]
            self.init_target = np.array([x, y])
            self.init_theta = np.random.uniform(-self.thetaMax, self.thetaMax, 2)
            '''通过极坐标方式在机械臂动作空间圆内生成随机目标点'''

        self.theta = self.init_theta.copy()
        self.omega = self.init_omega.copy()
        self.torque = np.zeros(2)
        self.target = self.init_target.copy()
        self.midPos = self.init_midPos.copy()
        self.endPos = self.init_endPos.copy()
        self.endVel = np.zeros(2)
        self.error = self.target - self.endPos
        self.time = 0.
        self.sum_d_theta = np.zeros(2)

        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()
        self.current_action = np.zeros(self.action_dim)
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0

        self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
        self.draw_init_image()
