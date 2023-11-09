import cv2 as cv
import numpy as np
import pandas as pd
from utils.functions import *
from utils.classes import Normalization
from algorithm.rl_base import rl_base
from environment.color import Color

class SecondOrderIntegration(rl_base):
    def __init__(self,
                 pos0: np.ndarray = np.array([1.0, 1.0]),
                 vel0: np.ndarray = np.array([0.0, 0.0]),
                 map_size: np.ndarray = np.array([5.0, 5.0]),
                 target: np.ndarray = np.array([2.5, 2.5])):
        super(SecondOrderIntegration, self).__init__()
        self.name = 'SecondOrderIntegration'
        self.init_pos = pos0
        self.init_vel = vel0
        self.map_size = map_size
        self.init_target = target

        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.acc = np.array([0., 0.])
        self.force = np.array([0., 0.])
        self.mass = 1.0
        self.target = self.init_target.copy()
        self.error = self.target - self.pos

        self.fMax = 3
        self.fMin = -3
        self.admissible_error = 5

        self.k = 0.15
        self.dt = 0.01  # 50Hz
        self.time = 0.  # time
        self.time_max = 5.0  # 每回合最大时间

        '''rl_base'''
        self.static_gain = 2
        self.state_dim = 4  # ex, ey, e_dx, e_dy
        self.state_num = [np.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.state_range = np.array(
            [[-self.map_size[0], self.map_size[0]],
             [-self.map_size[1], self.map_size[1]],
             [-np.inf, np.inf],
             [-np.inf, np.inf]])

        self.initial_state = self.get_state()
        self.current_state = self.get_state()
        self.next_state = self.get_state()

        self.action_dim = 2
        self.action_step = [None, None]
        self.action_range = [[self.fMin, self.fMax], [self.fMin, self.fMax]]

        self.action_num = [np.inf, np.inf]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.current_action = np.zeros(self.action_dim)

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        self.current_state_norm = Normalization(self.state_dim)
        self.next_state_norm = Normalization(self.state_dim)

        '''visualization'''
        self.x_offset = 20
        self.y_offset = 20
        self.board = 150
        self.pixel_per_meter = 50
        self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array(
            [self.x_offset, self.y_offset])).astype(int)
        self.image_size[0] += self.board
        self.image = 255 * np.ones([self.image_size[1], self.image_size[0], 3], np.uint8)
        self.image_white = self.image.copy()  # 纯白图
        self.draw_init_image()
        '''visualization'''

    def dis2pixel(self, coord) -> tuple:
        """
		:brief:         the transformation of coordinate between physical world and image
		:param coord:   position in physical world
		:return:        position in image coordinate
		"""
        x = self.x_offset + coord[0] * self.pixel_per_meter
        y = self.image_size[1] - self.y_offset - coord[1] * self.pixel_per_meter
        return int(x), int(y)

    def length2pixel(self, _l):
        """
		:brief:         the transformation of distance between physical world and image
		:param _l:      length in physical world
		:return:        length in image
		"""
        return int(_l * self.pixel_per_meter)

    def draw_ball(self):
        p_per_n = 0.6
        cv.circle(self.image, self.dis2pixel(self.pos), self.length2pixel(0.2), Color().Red, -1)  # 主体

        if self.force[0] > 0:
            cv.circle(self.image, self.dis2pixel(self.pos - np.array([0.25, 0])), self.length2pixel(0.1), Color().Blue,
                      -1)
            cv.line(self.image,
                    self.dis2pixel(self.pos - np.array([0.25, 0])),
                    self.dis2pixel(self.pos - np.array([0.25 + p_per_n * self.force[0], 0])),
                    Color().Blue, 2)
        elif self.force[0] < 0:
            cv.circle(self.image, self.dis2pixel(self.pos + np.array([0.25, 0])), self.length2pixel(0.1), Color().Blue,
                      -1)
            cv.line(self.image,
                    self.dis2pixel(self.pos + np.array([0.25, 0])),
                    self.dis2pixel(self.pos + np.array([0.25 - p_per_n * self.force[0], 0])),
                    Color().Blue, 2)
        else:
            pass

        if self.force[1] > 0:
            cv.circle(self.image, self.dis2pixel(self.pos - np.array([0., 0.25])), self.length2pixel(0.1), Color().Blue,
                      -1)
            cv.line(self.image,
                    self.dis2pixel(self.pos - np.array([0., 0.25])),
                    self.dis2pixel(self.pos - np.array([0., 0.25 + p_per_n * self.force[1]])),
                    Color().Blue, 2)
        elif self.force[1] < 0:
            cv.circle(self.image, self.dis2pixel(self.pos + np.array([0., 0.25])), self.length2pixel(0.1), Color().Blue,
                      -1)
            cv.line(self.image,
                    self.dis2pixel(self.pos + np.array([0., 0.25])),
                    self.dis2pixel(self.pos + np.array([0., 0.25 - p_per_n * self.force[1]])),
                    Color().Blue, 2)
        else:
            pass

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

    def draw_grid(self, num: np.ndarray = np.array([10, 10])):
        if np.min(num) <= 1:
            pass
        else:
            step = self.map_size / num
            for i in range(num[1] - 1):
                cv.line(self.image,
                        self.dis2pixel([0, 0 + (i + 1) * step[1]]),
                        self.dis2pixel([self.map_size[0], 0 + (i + 1) * step[1]]),
                        Color().Black, 1)
            for i in range(num[0] - 1):
                cv.line(self.image,
                        self.dis2pixel([0 + (i + 1) * step[0], 0]),
                        self.dis2pixel([0 + (i + 1) * step[0], self.map_size[1]]),
                        Color().Black, 1)

    def draw_init_image(self):
        self.draw_boundary()
        self.draw_target()
        self.draw_grid()
        self.image_white = self.image.copy()

    def visualization(self):
        self.image = self.image_white.copy()
        self.draw_ball()
        cv.putText(self.image, 'time:   %.3fs' % (round(self.time, 3)), (self.image_size[0] - self.board - 5, 25),
                   cv.FONT_HERSHEY_COMPLEX, 0.4, Color().Purple, 1)

        cv.putText(
            self.image,
            'pos: [%.2f, %.2f]m' % (round(self.pos[0], 3), round(self.pos[1], 3)),
            (self.image_size[0] - self.board - 5, 60), cv.FONT_HERSHEY_COMPLEX, 0.4, Color().Purple, 1)
        cv.putText(
            self.image,
            'error: [%.2f, %.2f]m' % (round(self.error[0], 3), round(self.error[1], 3)),
            (self.image_size[0] - self.board - 5, 95), cv.FONT_HERSHEY_COMPLEX, 0.4, Color().Purple, 1)
        cv.putText(
            self.image,
            'vel: [%.2f, %.2f]m/s' % (round(self.vel[0], 3), round(self.vel[1], 3)),
            (self.image_size[0] - self.board - 5, 140), cv.FONT_HERSHEY_COMPLEX, 0.4, Color().Purple, 1)

        cv.imshow(self.name, self.image)
        cv.waitKey(1)

    def get_state(self):
        self.error = self.target - self.pos
        return np.hstack((self.error, -self.vel))

    def is_out(self):
        right_out = self.pos[0] > self.map_size[0] + self.admissible_error
        left_out = self.pos[0] < 0 - self.admissible_error
        up_out = self.pos[1] > self.map_size[1] + self.admissible_error
        down_out = self.pos[1] < 0 - self.admissible_error
        if right_out or left_out or up_out or down_out:
            return True
        return False

    def is_success(self):
        if np.linalg.norm(self.error) <= 0.05 and np.linalg.norm(self.vel) < 0.05:
            return True
        return False

    def is_Terminal(self, param=None):
        self.is_terminal = False
        self.terminal_flag = 0
        if self.is_out():
            print('...out...')
            self.terminal_flag = 1
            self.is_terminal = True
        if self.time > self.time_max:
            # print('...time out...')
            self.terminal_flag = 2
            self.is_terminal = True
        # if self.is_success():
        #     print('...success...')
        #     self.terminal_flag = 3
        #     self.is_terminal = True

    def get_reward(self, param=None):
        # Q_pos = 0.1 * np.ones(2)
        # Q_vel = 0.01 * np.ones(2)
        # Q_acc = 0.001 * np.ones(2)
        #
        # e_pos = self.target - self.pos
        # e_vel = -self.vel
        #
        # u_pos = -np.dot(e_pos ** 2, Q_pos)
        # u_vel = -np.dot(e_vel ** 2, Q_vel)
        # u_acc = -np.dot(self.acc ** 2, Q_acc)
        Q_pos = 1
        Q_vel = 0.01
        Q_acc = 0.00

        e_pos = np.linalg.norm(self.target - self.pos)
        e_vel = np.linalg.norm(-self.vel)

        u_pos = -e_pos * Q_pos
        u_vel = -e_vel * Q_vel
        u_acc = -np.linalg.norm(self.acc) * Q_acc
        u_extra = 0.
        if e_pos < 2.5:
            u_pos += 2.0
        if self.terminal_flag == 1:     # position out
            _n = (self.time_max - self.time) / self.dt
            u_extra = _n * (u_pos + u_vel + u_acc)
        self.reward = u_pos + u_vel + u_acc + u_extra

    def ode(self, xx: np.ndarray):
        """
		@note:		注意，是微分方程里面的状态，不是 RL 的状态。
					xx = [x, y, vx, vy]，微分方程里面就这4个状态就可以
		@param xx:	state
		@return:	dx = f(x, t)，返回值当然是 \dot{xx}
		"""
        [_x, _y, _dx, _dy] = xx[:]
        _ddx = self.force[0] - self.k * _dx
        _ddy = self.force[1] - self.k * _dy
        return np.array([_dx, _dy, _ddx, _ddy])

    def rk44(self, action: np.ndarray):
        self.force = action.copy()
        self.acc = self.force / self.mass
        h = self.dt / 1
        tt = self.time + self.dt
        while self.time < tt:
            xx_old = np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1]])
            K1 = h * self.ode(xx_old)
            K2 = h * self.ode(xx_old + K1 / 2)
            K3 = h * self.ode(xx_old + K2 / 2)
            K4 = h * self.ode(xx_old + K3)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            [self.pos[0], self.pos[1], self.vel[0], self.vel[1]] = xx_new.copy()
            self.time += h

        self.acc = (self.force - self.k * self.vel) / self.mass
        self.error = self.target - self.pos

    def step_update(self, action: np.ndarray):
        """
		@param action:
		@return:
		"""
        self.current_action = action.copy()
        self.current_state = self.get_state()
        self.rk44(action=action)
        self.is_Terminal()
        self.next_state = self.get_state()
        self.get_reward()

    def reset(self, random: bool = False):
        if random:
            self.init_pos = np.array([np.random.uniform(0 + 0.1, self.map_size[0] - 0.1),
                                      np.random.uniform(0 + 0.1, self.map_size[1] - 0.1)])
            self.init_vel = np.zeros(2)
            self.init_target = self.map_size / 2
        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.acc = np.array([0., 0.])
        self.force = np.array([0., 0.])
        self.target = self.init_target.copy()
        self.error = self.target - self.pos
        self.time = 0.

        self.initial_state = self.get_state()
        self.current_state = self.get_state()
        self.next_state = self.get_state()
        self.current_action = np.zeros(self.action_dim)
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0

        self.image = 255 * np.ones([self.image_size[1], self.image_size[0], 3], np.uint8)
        self.image_white = self.image.copy()  # 纯白图
        self.draw_init_image()

    def save_state_norm(self, path, msg=None):
        data = {
            'cur_n': self.current_state_norm.running_ms.n * np.ones(self.state_dim),
            'cur_mean': self.current_state_norm.running_ms.mean,
            'cur_std': self.current_state_norm.running_ms.std,
            'cur_S': self.current_state_norm.running_ms.S,
            'next_n': self.next_state_norm.running_ms.n * np.ones(self.state_dim),
            'next_mean': self.next_state_norm.running_ms.mean,
            'next_std': self.next_state_norm.running_ms.std,
            'next_S': self.next_state_norm.running_ms.S,
        }
        if msg is None:
            pd.DataFrame(data).to_csv(path + 'state_norm.csv', index=False)
        else:
            pd.DataFrame(data).to_csv(path + 'state_norm_' + msg + '.csv', index=False)

    def load_norm_normalizer_from_file(self, path, file):
        data = pd.read_csv(path + file, header=0).to_numpy()
        self.current_state_norm.running_ms.n = data[0, 0]
        self.current_state_norm.running_ms.mean = data[:, 1]
        self.current_state_norm.running_ms.std = data[:, 2]
        self.current_state_norm.running_ms.S = data[:, 3]
        self.next_state_norm.running_ms.n = data[0, 4]
        self.next_state_norm.running_ms.mean = data[:, 5]
        self.next_state_norm.running_ms.std = data[:, 6]
        self.next_state_norm.running_ms.S = data[:, 7]
