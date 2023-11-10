import cv2 as cv
import numpy as np
from utils.functions import *
from algorithm.rl_base import rl_base
from environment.color import Color

class SecondOrderIntegration(rl_base):
    def __init__(self,
                 pos0: np.ndarray = np.array([5.0, 5.0]),
                 vel0: np.ndarray = np.array([0.0, 0.0]),
                 map_size: np.ndarray = np.array([10.0, 10.0]),
                 target: np.ndarray = np.array([5.0, 5.0]),
                 is_controller_BangBang: bool = False):
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

        self.vMax = 3
        self.vMin = -3
        self.fMax = 5
        self.fMin = -5
        self.aMax = self.fMax / self.mass
        self.aMin = self.fMin / self.mass

        self.k = 0.15
        self.dt = 0.02  # 50Hz
        self.time = 0.  # time
        self.timeMax = 5.0  # 每回合最大时间
        self.is_controller_BangBang = is_controller_BangBang  # 是否使用离散动作空间 BangBang 控制或者 Bang-3 控制

        '''rl_base'''
        self.use_normalization = True
        self.static_gain = 2
        self.state_dim = 6  # ex, ey, x, y, dx, dy
        self.state_num = [np.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.state_range = np.array(
            [[-self.map_size[0], self.map_size[0]],
             [-self.map_size[1], self.map_size[1]],
             [0, self.map_size[0]],
             [0, self.map_size[1]],
             [self.vMin, self.vMax],
             [self.vMin, self.vMax]]
        )
        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.hstack((self.error, self.pos, self.vel))
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 2
        self.action_step = [None, None]
        self.action_range = [[self.fMin, self.fMax], [self.fMin, self.fMax]]
        if self.is_controller_BangBang:
            self.action_num = [2, 2]
            self.action_space = [[self.fMin, self.fMax], [self.fMin, self.fMax]]
            self.isActionContinuous = [False, False]
        else:
            self.action_num = [np.inf, np.inf]
            self.action_space = [None, None]
            self.isActionContinuous = [True, True]
        self.initial_action = self.force.copy()
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        '''visualization'''
        self.x_offset = 20
        self.y_offset = 20
        self.board = 150
        self.pixel_per_meter = 50
        self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array(
            [self.x_offset, self.y_offset])).astype(int)
        self.image_size[0] += self.board
        self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8)* 255
        self.draw_boundary()
        self.draw_target()
        self.draw_grid()
        self.image_white = self.image.copy()
        '''visualization'''

    def state_norm(self):
        state = np.hstack((self.error, self.pos, self.vel))
        norm_min = self.state_range[:, 0]
        norm_max = self.state_range[:, 1]
        norm_s = (2 * state - (norm_min + norm_max)) / (norm_max - norm_min) * self.static_gain
        return norm_s

    def inverse_state_norm(self, s: np.ndarray):
        norm_min = self.state_range[:, 0]
        norm_max = self.state_range[:, 1]

        inverse_norm_s = (s * (norm_max - norm_min) / self.static_gain + (norm_min + norm_max)) / 2
        return inverse_norm_s

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

    def is_out(self):
        """简化处理，只判断中心的大圆有没有出界就好"""
        right_out = self.pos[0] > self.map_size[0]
        left_out = self.pos[0] < 0
        up_out = self.pos[1] > self.map_size[1]
        down_out = self.pos[1] < 0
        if right_out or left_out or up_out or down_out:
            return True
        return False

    def is_success(self):
        if np.linalg.norm(self.error) <= 0.05 and np.linalg.norm(self.vel) < 0.05:
            return True
        return False

    def is_Terminal(self, param=None):
        # if self.is_out():
        #     # print('...out...')
        #     self.terminal_flag = 1
        #     return True
        if self.time > self.timeMax:
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
        if self.is_controller_BangBang:
            if self.use_normalization:
                cur_s = self.inverse_state_norm(self.current_state)  # e, pos, vel
                nex_s = self.inverse_state_norm(self.next_state)
            else:
                cur_s = self.current_state
                nex_s = self.next_state

            cur_e = np.linalg.norm(cur_s[0: 2])
            nex_e = np.linalg.norm(nex_s[0: 2])

            nex_norm_e = nex_e / np.linalg.norm(self.map_size)
            # r1 = -nex_norm_e ** 2 * 1
            # if nex_norm_e < 0.25:
            # 	kk = -180 * nex_norm_e + 50
            # else:
            # 	kk = 5
            # r1 = (-nex_norm_e + 0.5) * kk

            # if cur_e > nex_e:
            # 	r1 = 2
            # else:
            # 	r1 = -2
            # r1 = - nex_e + 2  # for DQN demos
            # 出界反弹扣分
            # if (self.pos[0] > self.map_size[0] and self.vel[0] > 0 and self.current_action[0] > 0) or (
            #         self.pos[0] < 0 and self.vel[0] < 0 and self.current_action[0] < 0):
            #     r1 -= 20
            # if (self.pos[1] > self.map_size[1] and self.vel[1] > 0 and self.current_action[1] > 0) or (
            #         self.pos[1] < 0 and self.vel[1] < 0 and self.current_action[1] < 0):
            #     r1 -= 20
            r1 = -nex_norm_e - np.tanh(2.5 * nex_norm_e) + 1
            if self.terminal_flag == 3:  # 成功
                r4 = 1000
            elif self.terminal_flag == 2:  # 超时
                r4 = 0
            elif self.terminal_flag == 1:  # 出界
                r4 = -0
            else:
                r4 = 0

            self.reward = r1 + r4
        else:
            if self.use_normalization:
                cur_s = self.inverse_state_norm(self.current_state)  # e, pos, vel
                nex_s = self.inverse_state_norm(self.next_state)
            else:
                cur_s = self.current_state
                nex_s = self.next_state

            '''s1'''
            cur_error = np.linalg.norm(cur_s[0: 2])
            nex_error = np.linalg.norm(nex_s[0: 2])

            nex_vel = np.linalg.norm(nex_s[4: 6])

            R_e = 0.1
            # r1 = 0
            if cur_error > nex_error:  # 如果朝着正确的方向走
                r1 = np.linalg.norm(self.map_size) * R_e - nex_error ** 2 * R_e  # 位置二次型奖励
            else:
                r1 = 0
            # r1 = -nex_error ** 2 * R_e  # 位置二次型惩罚，走得越远罚得越多

            '''4. 其他'''
            if self.terminal_flag == 3:  # 成功
                r4 = 1000
            elif self.terminal_flag == 2:  # 超时
                r4 = 0
            elif self.terminal_flag == 1:  # 出界
                r4 = -0
            else:
                r4 = 0
            '''4. 其他'''
            # self.reward = rx_v + rx_e + ry_v + ry_e + r4		# s2
            # self.reward = r1 + r2 + r3 + r4					# s1
            yyf_x0 = nex_error / np.linalg.norm(self.map_size)
            if yyf_x0 < 0.25:
                kk = -180 * yyf_x0 + 50
            else:
                kk = 5
            r1 = (-yyf_x0 + 0.5) * kk
            # theta = np.arccos(
            #     np.dot(nex_s[4: 6], nex_s[0: 2]) / (np.linalg.norm(nex_s[0: 2]) * np.linalg.norm(nex_s[4: 6])))
            # if theta < rad2deg(45):  # 小于 45 度，不罚
            #     r2 = 0
            # else:
            #     r2 = -(theta - rad2deg(45)) * kk
            r2 = 0
            self.reward = r1 + r2 + r4

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
        self.vel[0] = np.clip(self.vel[0], self.vMin, self.vMax)
        self.vel[1] = np.clip(self.vel[1], self.vMin, self.vMax)
        self.acc = (self.force - self.k * self.vel) / self.mass
        self.error = self.target - self.pos

    def step_update(self, action: np.ndarray):
        """
		@param action:
		@return:
		"""
        self.current_action = action.copy()
        if self.use_normalization:
            self.current_state = self.state_norm()
        else:
            self.current_state = np.hstack((self.error, self.pos, self.vel))

        '''rk44'''
        self.rk44(action=action)  # 在微分方程里的，不在里面的，都更新了，角度也有判断
        '''rk44'''

        ''' 触界且动作仍然外界外推时使其反弹，法向速度为原来的80%防止出界 '''
        # if (self.pos[0] > self.map_size[0] and self.vel[0] > 0 and action[0] > 0) or (
        #         self.pos[0] < 0 and self.vel[0] < 0 and action[0] < 0):
        #     self.vel[0] *= -0.8
        # if (self.pos[1] > self.map_size[1] and self.vel[1] > 0 and action[1] > 0) or (
        #         self.pos[1] < 0 and self.vel[1] < 0 and action[1] < 0):
        #     self.vel[1] *= -0.8

        self.is_terminal = self.is_Terminal()
        if self.use_normalization:
            self.next_state = self.state_norm()
        else:
            self.next_state = np.hstack((self.error, self.pos, self.vel))

        self.get_reward()

        '''其他'''

        '''其他'''

    def reset(self):
        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.acc = np.array([0., 0.])
        self.force = np.array([0., 0.])
        self.target = self.init_target.copy()
        self.error = self.target - self.pos
        self.time = 0.

        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.hstack((self.error, self.pos, self.vel))
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.initial_action = self.force.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0

        self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
        self.draw_boundary()
        self.draw_target()
        self.draw_grid()
        self.image_white = self.image.copy()

    def reset_random(self):
        # self.init_pos = np.array([np.random.uniform(0 + 0.03, self.map_size[0] - 0.03),
        # 						  np.random.uniform(0 + 0.03, self.map_size[1] - 0.03)])

        self.init_pos = np.array([np.random.uniform(0 + 0.1, self.map_size[0] - 0.1),
                                  np.random.uniform(0 + 0.1, self.map_size[1] - 0.1)])

        self.init_vel = np.array([np.random.uniform(self.vMin, self.vMax),
                                  np.random.uniform(self.vMin, self.vMax)])
        self.init_target = self.map_size / 2

        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.acc = np.array([0., 0.])
        self.force = np.array([0., 0.])
        self.target = self.init_target.copy()
        self.error = self.target - self.pos
        self.time = 0.

        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.hstack((self.error, self.pos, self.vel))
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.initial_action = self.force.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0

        self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
        self.draw_boundary()
        self.draw_target()
        self.draw_grid()
        self.image_white = self.image.copy()
