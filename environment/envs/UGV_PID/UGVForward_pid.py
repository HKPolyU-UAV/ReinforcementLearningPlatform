from common.common_func import *
from environment.envs import *
from environment.envs.PIDControl.pid import PID


class UGV_Forward_PID(rl_base):
    def __init__(self,
                 pos0: np.ndarray = np.array([5.0, 5.0]),
                 vel0: np.ndarray = np.array([0., 0.]),
                 phi0: float = 0.,
                 omega0: float = 0.,
                 map_size: np.ndarray = np.array([10.0, 10.0]),
                 target: np.ndarray = np.array([5.0, 5.0])):
        super(UGV_Forward_PID, self).__init__()

        self.init_pos = pos0
        self.init_vel = vel0
        self.init_phi = phi0
        self.init_omega = omega0
        self.init_target = target

        self.pos = pos0
        self.vel = vel0
        self.phi = phi0
        self.omega = omega0
        self.map_size = map_size
        self.target = target
        self.error = self.target - self.pos
        self.w_wheel = np.zeros(2, dtype=np.float32)  # 车轮角速度
        self.a_wheel = np.zeros(2, dtype=np.float32)  # 车轮角加速度

        self.pid = np.zeros(4, dtype=np.float32)  # pd四个参数
        self.posPID = PID()  # 位置PD
        self.thetaPID = PID()  # 角度PD

        '''hyper-parameters'''
        self.r = 0.1  # 车轮半径
        self.l_wheel = 0.06  # 车轮厚度
        self.rBody = 0.15  # 车主体半径
        self.L = 2 * self.rBody  # 车主体直径
        self.dt = 0.02  # 50Hz
        self.time = 0.  # time
        self.timeMax = 8.0
        '''hyper-parameters'''

        self.posPMax = 300.0  # PD参数范围
        self.posPMin = 5.0
        self.posDMax = 5.0  # PD参数范围
        self.posDMin = 0.0
        self.thetaPMax = 500.0  # PD参数范围
        self.thetaPMin = 100.0
        self.thetaDMax = 5.0  # PD参数范围
        self.thetaDMin = 0.0

        self.wMax = 20  # 车轮最大角速度   rad/s
        self.wMin = 0
        self.aMax = 100  # 车轮最大角加速度 rad/s^2
        self.aMin = -100
        self.phiMax = deg2rad(180)  # 车体最大角度
        self.phiMin = -deg2rad(180)
        self.vMax = self.r * self.wMax  # 2 m/s
        self.vMin = self.r * self.wMin  # -2 m/s
        self.omegaMax = self.r / self.L * (self.wMax - self.wMin)  # 车体最大角速度
        self.omegaMin = self.r / self.L * (self.wMin - self.wMax)

        self.miss = self.rBody  # 容许误差
        self.name = 'UGVBidirectional_pid'

        '''rl_base'''
        self.use_normalization = True
        self.static_gain = 2
        self.state_dim = 5  # ex, ey, wl, wr, dphi 位置误差，线速度，角度，角速度
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.state_range = np.array(
            [[-self.map_size[0], self.map_size[0]],
             [-self.map_size[1], self.map_size[1]],
             [self.wMin, self.wMax],
             [self.wMin, self.wMax],
             [self.omegaMin, self.omegaMax]]
        )
        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.append(self.error, [self.w_wheel[0], self.w_wheel[1], self.omega])
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 4  # 位置PD和角度PD
        self.action_step = [None, None]
        self.action_range = np.array(
            [[self.posPMin, self.posPMax],
             [self.posDMin, self.posDMax],
             [self.thetaPMin, self.thetaPMax],
             [self.thetaDMin, self.thetaDMax]]
        )
        self.action_num = [math.inf, math.inf]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.initial_action = self.pid.copy()
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        '''visualization'''
        self.x_offset = 20
        self.y_offset = 20
        self.board = 170
        self.pixel_per_meter = 50
        self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array(
            [self.x_offset, self.y_offset])).astype(int)
        self.image_size[0] += self.board
        self.image = np.zeros([self.image_size[1], self.image_size[0], 3], np.uint8)
        self.image[:, :, 0] = np.ones([self.image_size[1], self.image_size[0]]) * 255
        self.image[:, :, 1] = np.ones([self.image_size[1], self.image_size[0]]) * 255
        self.image[:, :, 2] = np.ones([self.image_size[1], self.image_size[0]]) * 255
        self.image_white = self.image.copy()  # 纯白图
        '''visualization'''

        self.sum_d_theta = 0.
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

    def show_dynamic_image(self, isWait=False):
        self.image = self.image_white.copy()
        self.draw_boundary()
        self.draw_car()
        self.draw_target()
        self.draw_grid()

        cv.putText(
            self.image,
            'time:   %.3fs' % (round(self.time, 3)),
            (self.image_size[0] - self.board - 5, 25), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

        cv.putText(
            self.image,
            'pos: [%.2f, %.2f]m' % (round(self.pos[0], 3), round(self.pos[1], 3)),
            (self.image_size[0] - self.board - 5, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'error: [%.2f, %.2f]m' % (round(self.error[0], 3), round(self.error[1], 3)),
            (self.image_size[0] - self.board - 5, 95), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'vel: %.2fm/s' % (round((self.w_wheel[0] + self.w_wheel[1]) / 2, 2)),
            (self.image_size[0] - self.board - 5, 130), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'phi: %.2f ' % (round(rad2deg(self.phi), 2)),
            (self.image_size[0] - self.board - 5, 165), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'omega: %.2f PI' % (round(self.omega / np.pi, 2)),
            (self.image_size[0] - self.board - 5, 200), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

        cv.imshow(self.name, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)

    def draw_car(self):
        rBody = self.rBody * 3
        r = self.r * 2
        l_wheel = self.l_wheel * 2
        cv.circle(self.image, self.dis2pixel(self.pos), self.length2pixel(self.rBody), Color().Orange, -1)  # 主体
        '''两个车轮'''
        # left
        pts_left = [[r, rBody], [r, rBody - l_wheel], [-r, rBody - l_wheel], [-r, rBody]]
        pts_left = points_rotate(pts_left, self.phi)
        pts_left = points_move(pts_left, [self.pos[0], self.pos[1]])
        cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in pts_left]), color=Color().Red)
        # right
        pts_right = [[r, -rBody], [r, -rBody + l_wheel], [-r, -rBody + l_wheel], [-r, -rBody]]
        pts_right = points_rotate(pts_right, self.phi)
        pts_right = points_move(pts_right, [self.pos[0], self.pos[1]])
        cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in pts_right]),
                          color=Color().Red)
        '''两个车轮'''
        # 额外画一个圆形，标志头
        line = points_move(points_rotate([[self.rBody, 0], [3 * self.rBody, 0]], self.phi), self.pos)
        cv.line(self.image, self.dis2pixel(line[0]), self.dis2pixel(line[1]), Color().Blue, 2)
        ball = points_move(points_rotate([3 * self.rBody, 0], self.phi), self.pos)
        cv.circle(self.image, self.dis2pixel(ball), self.length2pixel(0.08), Color().Red, -1)  # 主体

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

    def state_norm(self):
        state = np.append(self.error, [self.w_wheel[0], self.w_wheel[1], self.omega])
        norm_min = self.state_range[:, 0]
        norm_max = self.state_range[:, 1]
        norm_s = (2 * state - (norm_min + norm_max)) / (norm_max - norm_min) * self.static_gain
        return norm_s

    def inverse_state_norm(self, s: np.ndarray):
        norm_min = self.state_range[:, 0]
        norm_max = self.state_range[:, 1]

        inverse_norm_s = (s * (norm_max - norm_min) / self.static_gain + (norm_min + norm_max)) / 2
        return inverse_norm_s

    def is_out(self):
        """
		:return:
		"""
        '''简化处理，只判断中心的大圆有没有出界就好'''
        right_out = self.pos[0] + self.rBody > self.map_size[0]
        left_out = self.pos[0] - self.rBody < 0
        up_out = self.pos[1] + self.rBody > self.map_size[1]
        down_out = self.pos[1] - self.rBody < 0
        if right_out or left_out or up_out or down_out:
            return True
        return False

    def is_success(self):
        b1 = np.linalg.norm(self.error) <= self.miss
        b2 = np.fabs(self.omega) < deg2rad(5)
        b3 = np.linalg.norm(self.vel) < 0.02
        return b1 and b2 and b3

    def is_Terminal(self, param=None):
        if self.time > self.timeMax:
            # print('...time out...')
            self.terminal_flag = 2
            return True
        if self.is_success():
            # print('...success...')
            self.terminal_flag = 3
            return False
        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        if self.use_normalization:
            cur_s = self.inverse_state_norm(self.current_state)
            nex_s = self.inverse_state_norm(self.next_state)
        else:
            cur_s = self.current_state
            nex_s = self.next_state

        cur_error = np.linalg.norm(cur_s[0: 2])
        nex_error = np.linalg.norm(nex_s[0: 2])

        cur_norm_error = cur_error / np.linalg.norm(self.map_size)
        nex_norm_error = nex_error / np.linalg.norm(self.map_size)

        cur_phi = cur_s[-2]
        cur_v_head = np.array([np.cos(cur_phi), np.sin(cur_phi)])
        cur_error_theta = np.arccos(np.dot(cur_s[0: 2], cur_v_head) / np.linalg.norm(cur_s[0: 2]))
        cur_error_theta = min(cur_error_theta, np.pi - cur_error_theta)
        cur_norm_error_theta = cur_error_theta / np.pi

        nex_phi = nex_s[-2]
        nex_v_head = np.array([np.cos(nex_phi), np.sin(nex_phi)])
        nex_error_theta = np.arccos(np.dot(nex_s[0: 2], nex_v_head) / np.linalg.norm(nex_s[0: 2]))
        nex_error_theta = min(nex_error_theta, np.pi - nex_error_theta)
        nex_norm_error_theta = nex_error_theta / np.pi

        if self.sum_d_theta > 4 * np.pi:  # 如果转的超过两圈
            self.terminal_flag = 4

        '''4. 其他'''
        if self.terminal_flag == 4:  # 瞎几把转
            r4 = -0
        elif self.terminal_flag == 3:  # 成功
            r4 = 5
        elif self.terminal_flag == 2:  # 超时
            r4 = -0
        elif self.terminal_flag == 1:  # 出界
            r4 = -0
        else:
            r4 = 0
        '''4. 其他'''

        '''r1 是位置'''
        r1 = -nex_norm_error - np.tanh(5 * nex_norm_error) + 1
        '''r2 是角度'''
        if nex_error < 4 * self.miss:  # 如果误差比较小，就不考虑角度了
            r2 = 0
        else:
            r2 = -nex_norm_error_theta - np.tanh(2.5 * nex_norm_error_theta) + 1
        r2 = 0
        self.reward = r1 + r2 + r4

    def ode(self, xx: np.ndarray):
        [_x, _y, _phi, _wl, _wr] = xx[:]
        _wl = np.clip(_wl, self.wMin, self.wMax)
        _wr = np.clip(_wr, self.wMin, self.wMax)
        _dx = self.r / 2 * (_wl + _wr) * np.cos(_phi)
        _dy = self.r / 2 * (_wl + _wr) * np.sin(_phi)
        _dphi = self.r / self.L * (_wr - _wl)
        _dwl = self.a_wheel[0]
        _dwr = self.a_wheel[1]
        return np.array([_dx, _dy, _dphi, _dwl, _dwr])

    def rk44(self, action: np.ndarray):
        self.a_wheel = np.clip(action.copy(), self.aMin, self.aMax)
        h = self.dt / 1
        tt = self.time + self.dt
        phi = self.phi
        while self.time < tt:
            xx_old = np.array([self.pos[0], self.pos[1], self.phi, self.w_wheel[0], self.w_wheel[1]])
            K1 = h * self.ode(xx_old)
            K2 = h * self.ode(xx_old + K1 / 2)
            K3 = h * self.ode(xx_old + K2 / 2)
            K4 = h * self.ode(xx_old + K3)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            xx_new[3] = np.clip(xx_new[3], self.wMin, self.wMax)
            xx_new[4] = np.clip(xx_new[4], self.wMin, self.wMax)
            [self.pos[0], self.pos[1], self.phi, self.w_wheel[0], self.w_wheel[1]] = xx_new.copy()
            self.time += h
        self.vel[0] = self.r / 2 * (self.w_wheel[0] + self.w_wheel[1]) * np.cos(self.phi)
        self.vel[1] = self.r / 2 * (self.w_wheel[0] + self.w_wheel[1]) * np.sin(self.phi)
        self.omega = self.r / self.L * (self.w_wheel[1] - self.w_wheel[0])
        self.error = self.target - self.pos
        self.sum_d_theta += np.fabs(phi - self.phi)
        if self.phi > self.phiMax:
            self.phi -= 2 * self.phiMax
        if self.phi < self.phiMin:
            self.phi -= 2 * self.phiMin

    def step_update(self, action: np.ndarray):
        self.current_action = action.copy()
        if self.use_normalization:
            self.current_state = self.state_norm()
        else:
            self.current_state = np.append(self.error, [self.w_wheel[0], self.w_wheel[1], self.omega])

        '''PD'''
        posError = self.target - self.pos
        thetaError = np.arctan2(posError[1], posError[0]) - self.phi
        self.posPID.set_pid(self.current_action[0], 0, self.current_action[1])
        self.posPID.set_e(np.linalg.norm(posError))
        self.thetaPID.set_pid(self.current_action[2], 0, self.current_action[3])
        self.thetaPID.set_e(thetaError)
        '''PD'''

        '''rk44'''
        wRef = np.array([self.posPID.out() - self.rBody * self.thetaPID.out(), self.posPID.out() + self.rBody * self.thetaPID.out()])
        self.rk44(action=(wRef - self.w_wheel) / self.dt)
        '''rk44'''

        self.is_terminal = self.is_Terminal()

        if self.use_normalization:
            self.next_state = self.state_norm()
        else:
            self.next_state = np.append(self.error, [self.w_wheel[0], self.w_wheel[1], self.omega])
        self.get_reward()

        '''其他'''

        '''其他'''

    def reset(self):
        self.pos = self.init_pos.copy()
        self.vel = self.init_vel
        self.phi = self.init_phi
        self.omega = self.init_omega
        self.target = self.init_target.copy()
        self.error = self.target - self.pos
        self.pid = np.zeros(4, dtype=np.float32)
        self.posPID.reset()
        self.thetaPID.reset()
        self.time = 0.
        self.sum_d_theta = 0.
        self.w_wheel = np.zeros(2)
        self.a_wheel = np.zeros(2)

        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.append(self.error, [self.w_wheel[0], self.w_wheel[1], self.omega])
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.initial_action = self.pid.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功

    def reset_random(self):
        # self.init_pos = np.array([np.random.uniform(0 + self.rBody + 0.03, self.map_size[0] - self.rBody - 0.03),
        # 						  np.random.uniform(0 + self.rBody + 0.03, self.map_size[1] - self.rBody - 0.03)])
        self.init_pos = self.map_size / 2.0
        self.init_phi = np.random.uniform(self.phiMin, self.phiMax)
        # self.init_phi = 0.

        '''给随机初始的轮速'''
        # self.w_wheel = np.array([np.random.uniform(self.wMin, self.wMax),np.random.uniform(self.wMin, self.wMax)])
        self.w_wheel = np.array([0., 0.])
        '''计算初始速度'''
        self.init_vel[0] = self.r / 2 * (self.w_wheel[0] + self.w_wheel[1]) * np.cos(self.init_phi)
        self.init_vel[1] = self.r / 2 * (self.w_wheel[0] + self.w_wheel[1]) * np.sin(self.init_phi)

        '''计算初始角速度'''
        self.init_omega = 0.
        self.init_target = np.array([np.random.uniform(0 + self.rBody + 0.1, self.map_size[0] - self.rBody - 0.1),
                                     np.random.uniform(0 + self.rBody + 0.1, self.map_size[1] - self.rBody - 0.1)])

        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.phi = self.init_phi
        self.omega = self.init_omega
        self.target = self.init_target.copy()
        self.error = self.target - self.pos
        self.pid = np.zeros(4, dtype=np.float32)
        self.posPID.reset()
        self.thetaPID.reset()
        self.time = 0.
        self.sum_d_theta = 0.
        self.a_wheel = np.zeros(2)

        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.append(self.error, [self.w_wheel[0], self.w_wheel[1], self.omega])
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.initial_action = self.pid.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
