import cv2 as cv
import numpy as np

from utils.functions import *
from algorithm.rl_base import rl_base
from environment.color import Color
from map import Map


class UGVForwardObstacleAvoidance(rl_base, Map):
    def __init__(self,
                 pos0: np.ndarray = np.array([1., 1.]),
                 phi0: float = 0.,
                 map_size: np.ndarray = np.array([5.0, 5.0]),
                 target: np.ndarray = np.array([2.5, 2.5])):
        """
        Args:
            pos0:
            phi0:
            map_size:
            target:
        """
        rl_base.__init__(self)
        Map.__init__(self)

        self.init_pos = pos0
        self.init_phi = phi0
        self.init_target = target

        self.pos = pos0
        self.vel = 0.
        self.phi = phi0
        self.omega = 0.
        self.map_size = map_size
        self.target = target
        self.error = self.get_e()
        self.e_phi = self.get_e_phi()

        self.r_vehicle = 0.15  # 车的半径

        '''hyper-parameters'''
        self.dt = 0.02  # 50Hz
        self.time = 0.  # time
        self.time_max = 10.0  # 每回合最大时间
        self.a_linear = 0.  # 等效线加速度
        self.a_angular = 0.  # 等效角加速度
        self.kf = 0.1  # 等效线阻力系数
        self.kt = 0.1  # 等效角阻力系数

        self.laserDis = 2.0  # 雷达探测半径
        self.laserBlind = 0.0  # 雷达盲区
        self.laserRange = deg2rad(90)  # 左右各90度，一共180度
        self.laserStep = deg2rad(5)
        self.laserState = int(2 * self.laserRange / self.laserStep) + 1  # 雷达的线数
        self.visualLaser = np.zeros((self.laserState, 2))  # 一行一个探测点坐标
        self.visualFlag = np.zeros(self.laserState)
        '''hyper-parameters'''

        '''state limitation'''
        self.e_max = np.linalg.norm(self.map_size) / 2
        self.v_max = 3
        self.e_phi_max = np.pi
        self.omega_max = 2 * np.pi
        self.a_linear_max = 3
        self.a_angular_max = 2 * np.pi
        '''state limitation'''

        self.name = 'UGVForwardObstacleAvoidance'

        '''rl_base'''
        self.use_norm = True
        self.static_gain = 1.
        self.state_dim = 4 + self.laserState  # e, v, e_theta, omega 位置误差，线速度，角度误差，角速度
        self.state_num = [np.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        if self.use_norm:
            self.state_range = np.array([-self.static_gain, self.static_gain] for _ in range(self.state_dim))
        else:
            self.state_range = np.concatenate((np.array(
                [[0, self.e_max],
                 [0, self.v_max],
                 [-self.e_phi_max, self.e_phi_max],
                 [-self.omega_max, self.omega_max]]
            ), self.laserDis * np.ones(self.laserState)))
        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()

        self.action_dim = 2
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_range = np.array(
            [[-self.a_linear_max, self.a_linear_max], [-self.a_angular_max, self.a_angular_max]])
        self.action_num = [np.inf for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.isActionContinuous = [True for _ in range(self.action_dim)]
        self.current_action = np.array([self.a_linear, self.a_angular])

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        self.obs = []
        self.obsNum = 0

        '''visualization'''
        self.x_offset = 20
        self.y_offset = 20
        self.board = 170
        self.pixel_per_meter = 80
        self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array(
            [self.x_offset, self.y_offset])).astype(int)
        self.image_size[0] += self.board
        self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        '''visualization'''

        self.reset(True)

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

    def draw_grid(self):
        xNum = 5
        yNum = 5
        stepy = self.map_size[0] / xNum
        for i in range(xNum):
            cv.line(self.image,
                    self.dis2pixel([0, 0 + (i + 1) * stepy]),
                    self.dis2pixel([self.map_size[0], 0 + (i + 1) * stepy]),
                    Color().Black, 1)
        stepx = self.map_size[1] / yNum
        for i in range(yNum):
            cv.line(self.image,
                    self.dis2pixel([0 + (i + 1) * stepx, 0]),
                    self.dis2pixel([0 + (i + 1) * stepx, self.map_size[1]]),
                    Color().Black, 1)

    def draw_obs(self):
        for [name, c, constraints] in self.obs:
            color = Color().DarkGray
            if name == 'circle':
                cv.circle(self.image, self.dis2pixel(c), self.length2pixel(constraints[0]), color, -1)
            elif name == 'ellipse':
                cv.ellipse(img=self.image,
                           center=self.dis2pixel(c),
                           axes=(self.length2pixel(constraints[0]), self.length2pixel(constraints[1])),
                           angle=-constraints[2],
                           startAngle=0.,
                           endAngle=360.,
                           color=color,
                           thickness=-1)
            else:
                cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in c]),
                                  color=color)

    def map_draw_photo_frame(self):
        cv.rectangle(self.image, (0, 0), (self.image_size[0] - 1, self.dis2pixel([self.map_size[0], self.map_size[1]])[1]), Color().White, -1)
        cv.rectangle(self.image, (0, 0), (self.dis2pixel([0., 0.])[0], self.image_size[1] - 1), Color().White, -1)
        cv.rectangle(self.image, self.dis2pixel([self.map_size[0], self.map_size[1]]), (self.image_size[0] - 1, self.image_size[1] - 1), Color().White, -1)
        cv.rectangle(self.image, self.dis2pixel([0., 0.]), (self.image_size[0] - 1, self.image_size[1] - 1), Color().White, -1)

    def draw_init_image(self):
        self.draw_obs()
        self.map_draw_photo_frame()
        self.draw_boundary()
        self.draw_grid()
        self.image_copy = self.image.copy()

    def draw_text(self):
        cv.putText(
            self.image,
            'time: %.2f s' % (round(self.time, 2)),
            (self.image_size[0] - self.board - 5, 25), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'pos: [%.2f, %.2f] m' % (round(self.pos[0], 3), round(self.pos[1], 3)),
            (self.image_size[0] - self.board - 5, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'e_pos: %.2f m' % (round(self.error, 2)),
            (self.image_size[0] - self.board - 5, 95), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'vel: %.2f m/s' % (round(self.vel, 2)),
            (self.image_size[0] - self.board - 5, 130), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'phi: %.2f ' % (round(rad2deg(self.phi), 2)),
            (self.image_size[0] - self.board - 5, 165), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'e_phi: %.2f ' % (round(rad2deg(self.e_phi), 2)),
            (self.image_size[0] - self.board - 5, 200), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'omega: %.2f PI' % (round(self.omega / np.pi, 2)),
            (self.image_size[0] - self.board - 5, 235), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

    def draw_target(self):
        cv.circle(self.image, self.dis2pixel(self.target), 5, Color().random_color_by_BGR(), -1)

    def draw_car(self):
        _l = self.r_vehicle * 2.5
        pt1 = self.dis2pixel(self.pos)
        pt2 = self.dis2pixel(self.pos + np.array([_l * np.cos(self.phi), _l * np.sin(self.phi)]))
        cv.circle(self.image, pt1, self.length2pixel(self.r_vehicle), Color().Orange, -1)
        cv.line(self.image, pt1, pt2, Color().Blue, 2)
        cv.circle(self.image, pt2, 3, Color().Red, -1)

    def draw_laser(self):
        for _laser, _flag in zip(self.visualLaser, self.visualFlag):
            if _flag == 0:
                cv.circle(self.image, self.dis2pixel(_laser), self.length2pixel(0.04), Color().Purple, -1)  # 啥也没有
            elif _flag == 1:
                cv.circle(self.image, self.dis2pixel(_laser), self.length2pixel(0.04), Color().LightPink, -1)  # 有东西
            else:
                cv.circle(self.image, self.dis2pixel(_laser), self.length2pixel(0.04), Color().Red, -1)  # 盲区

    def visualization(self):
        self.image = self.image_copy.copy()
        self.draw_car()
        self.draw_target()
        self.draw_laser()
        self.draw_text()
        cv.imshow(self.name, self.image)
        cv.waitKey(1)

    def collision_check(self):
        for _obs in self.obs:
            if _obs[0] == 'circle':
                if dis_two_points(self.pos, _obs[1]) <= _obs[2][0] + self.r_vehicle:
                    return True
            elif _obs[0] == 'ellipse':
                print('In function: <collision_check> ERROR!!!')
                return False
            else:
                print('In function: <collision_check> ERROR!!!')
                return False
        return False

    def get_fake_laser(self) -> np.ndarray:
        laser = []
        detectPhi = np.linspace(self.phi - self.laserRange, self.phi + self.laserRange, self.laserState)  # 所有的角度
        count = 0

        x = self.pos[0]
        y = self.pos[1]
        xm = self.map_size[0]
        ym = self.map_size[1]

        '''如果车本身在障碍物里面'''
        if self.collision_check():
            for i in range(self.laserState):
                laser.append(self.laserBlind)
                self.visualLaser[i] = [x, y]
                self.visualFlag[i] = 1
            return np.array(laser)
        '''如果车本身在障碍物里面'''

        start = np.array([x, y])
        '''1. 提前求出起点与障碍物中心距离，然后将距离排序'''
        ref_dis = []
        for _obs in self.obs:
            if _obs[0] == 'circle':
                ref_dis.append(dis_two_points([x, y], _obs[1]))
            else:
                ref_dis.append(dis_point_2_poly(_obs[1], [x, y]))
        ref_sort = np.argsort(ref_dis)  # 排序的障碍物，距离从小到达，越小的说明离机器人越近
        '''1. 提前求出起点与障碍物中心距离，然后将距离排序'''
        for phi in detectPhi:
            if phi > np.pi:
                phi -= 2 * np.pi
            if phi < -np.pi:
                phi += 2 * np.pi
            m = np.tan(phi)  # 斜率
            b = y - m * x  # 截距

            '''2. 确定当前机器人与四个角点的连接'''
            theta1 = cal_vector_rad([1, 0], [xm - x, ym - y])  # 右上
            theta2 = cal_vector_rad([1, 0], [0 - x, ym - y])  # 左上
            theta3 = -cal_vector_rad([1, 0], [0 - x, 0 - y])  # 左下
            theta4 = -cal_vector_rad([1, 0], [xm - x, 0 - y])  # 右下
            '''2. 确定当前机器人与四个角点的连接'''

            '''3. 找到终点'''
            cosTheta = np.fabs(m) / np.sqrt(1 + m ** 2)
            sinTheta = 1 / np.sqrt(1 + m ** 2)
            if theta4 < phi <= theta1:
                terminal = [xm, m * xm + b]
                tx = x + self.laserDis / np.sqrt(1 + m ** 2)
                if tx < xm:
                    terminal = [tx, y + cosTheta * self.laserDis] if m >= 0 else [tx, y - cosTheta * self.laserDis]
            elif theta1 < phi <= theta2:
                terminal = [(ym - b) / m, ym] if np.fabs(m) < 1e8 else [x, ym]
                ty = y + np.fabs(m) * self.laserDis / np.sqrt(1 + m ** 2)
                if ty < ym:
                    terminal = [x + self.laserDis * sinTheta, ty] if m >= 0 else [x - self.laserDis * sinTheta, ty]
            elif theta3 < phi <= theta4:
                terminal = [-b / m, 0] if np.fabs(m) < 1e8 else [x, 0]
                ty = y - np.fabs(m) * self.laserDis / np.sqrt(1 + m ** 2)
                if ty > 0:
                    terminal = [x - self.laserDis * sinTheta, ty] if m >= 0 else [x + self.laserDis * sinTheta, ty]
            else:
                terminal = [0, b]
                tx = x - self.laserDis / np.sqrt(1 + m ** 2)
                if tx > 0:
                    terminal = [tx, y - cosTheta * self.laserDis] if m >= 0 else [tx, y + cosTheta * self.laserDis]
            terminal = np.array(terminal)
            '''3. 找到终点'''

            '''4. 开始找探测点'''
            find = False
            for index in ref_sort:
                _obs = self.obs[index]
                if _obs[0] == 'circle':  # 如果障碍物是圆
                    x0 = _obs[1][0]
                    y0 = _obs[1][1]
                    r0 = _obs[2][0]
                    if ref_dis[index] > self.laserDis + r0:
                        continue  # 如果障碍物本身超出可探测范围，那么肯定不用考虑
                    if np.fabs(m * x0 - y0 + b) / np.sqrt(1 + m ** 2) > r0:
                        continue  # 如果圆心到线段所在直线的距离大于圆的半径，那么肯定不用考虑
                    if cal_vector_rad([terminal[0] - start[0], terminal[1] - start[1]], [x0 - start[0], y0 - start[1]]) > np.pi / 2:
                        continue  # 如果圆心的位置在探测线段的后方，那么肯定是不需要考虑
                    '''能执行到这，就说明存在一个园，使得线段所在的射线满足条件，只需要计算点是否在线段上即可'''
                    # 垂足坐标
                    foot_x = (x0 + m * y0 - m * b) / (m ** 2 + 1)
                    foot_y = (m * x0 + m ** 2 * y0 + b) / (m ** 2 + 1)
                    r_dis = dis_two_points(np.array([foot_x, foot_y]), np.array([x0, y0]))
                    # dis_slide = math.sqrt(r0 ** 2 - r_dis ** 2)     # 垂足到交点滑动距离
                    crossPtx = foot_x - np.sign(terminal[0] - start[0]) * np.sqrt(r0 ** 2 - r_dis ** 2) / np.sqrt(m ** 2 + 1)
                    if min(start[0], terminal[0]) <= crossPtx <= max(start[0], terminal[0]):
                        find = True
                        dis = np.fabs(crossPtx - start[0]) * np.sqrt(m ** 2 + 1)
                        if dis < self.laserBlind:  # too close
                            laser.append(self.laserBlind)
                            newX = start[0] + self.laserBlind / np.sqrt(m ** 2 + 1) * np.sign(terminal[0] - start[0])
                            self.visualLaser[count] = [newX, m * newX + b]
                            self.visualFlag[count] = 2
                        else:
                            laser.append(dis)
                            self.visualLaser[count] = [crossPtx, m * crossPtx + b]
                            self.visualFlag[count] = 1
                        break
                else:  # 如果障碍物是多边形    TODO 留给读者思考，参考之前的
                    pass
            '''4. 开始找探测点'''

            if not find:  # 点一定是终点，但是属性不一定
                dis = dis_two_points(start, terminal)
                if dis > self.laserDis:  # 如果起始点与终点的距离大于探测半径，那么就直接给探测半径，相当于空场地
                    laser.append(self.laserDis)
                    self.visualLaser[count] = terminal.copy()
                    self.visualFlag[count] = 0
                elif self.laserBlind < dis <= self.laserDis:  # 如果起始点与终点的距离小于探测半径，那么直接给距离，说明探测到场地边界
                    laser.append(dis)
                    self.visualLaser[count] = terminal.copy()
                    self.visualFlag[count] = 0
                else:  # 进入雷达盲区，0m
                    laser.append(self.laserBlind)
                    self.visualLaser[count] = terminal.copy()
                    self.visualFlag[count] = 2
            count += 1
        return np.array(laser)

    def get_state(self) -> np.ndarray:
        self.error = self.get_e()
        self.e_phi = self.get_e_phi()
        laser = self.get_fake_laser()
        if self.use_norm:
            _s = 2 / self.e_max * self.error - 1
            _vel = 2 / self.v_max * self.vel - 1
            _e_phi = self.e_phi / self.e_phi_max
            _omega = self.omega / self.omega_max
            _laser = 2 * laser / self.laserDis - 1
            return np.concatenate(([_s, _vel, _e_phi, _omega], _laser)) * self.static_gain
        else:
            return np.concatenate(([self.error, self.vel, self.e_phi, self.omega], laser))

    def is_out(self):
        """
        :return:
        """
        right_out = self.pos[0] > self.map_size[0]
        left_out = self.pos[0] < 0
        up_out = self.pos[1] > self.map_size[1]
        down_out = self.pos[1] < 0
        return right_out or left_out or up_out or down_out

    def is_success(self):
        b1 = np.fabs(self.error) <= 0.05
        b2 = np.fabs(self.omega) < 0.01
        # b2 = True
        b3 = np.fabs(self.vel) < 0.01

        return b1 and b2 and b3

    def is_Terminal(self, param=None):
        self.terminal_flag = 0
        self.is_terminal = False
        if self.is_out():
            # print('...out...')
            self.terminal_flag = 1
            self.is_terminal = True
        if self.time > self.time_max:
            # print('...time out...')
            self.terminal_flag = 2
            self.is_terminal = True
        if self.is_success():
            print('...success...')
            self.terminal_flag = 3
            self.is_terminal = True
        if self.collision_check():
            print('...collision...')
            self.terminal_flag = 4
            self.is_terminal = True

    def get_reward(self, param=None):
        Q_pos = 2.
        Q_vel = 0.0
        Q_phi = 2.
        Q_omega = 1.0

        u_pos = -np.fabs(self.error) * Q_pos
        u_vel = -np.fabs(self.vel) * Q_vel
        u_phi = -np.fabs(self.e_phi) * Q_phi if self.error > 0.1 else 0.0
        u_omega = -np.fabs(self.omega) * Q_omega

        u_psi = 0.
        if self.terminal_flag == 1:  # 出界
            _n = (self.time_max - self.time) / self.dt
            u_psi = _n * (u_pos + u_vel + u_phi + u_omega)

        self.reward = u_pos + u_vel + u_phi + u_omega + u_psi

    def ode(self, xx: np.ndarray):
        """
        @param xx:	state
        @return:	dx = f(x, t)，返回值当然是  dot{xx}
        """
        [_x, _y, _vel, _phi, _omega] = xx[:]
        _dx = _vel * np.cos(_phi)
        _dy = _vel * np.sin(_phi)
        _dvel = self.a_linear - self.kf * _vel
        _dphi = _omega
        _domega = self.a_angular - self.kt * _omega
        return np.array([_dx, _dy, _dvel, _dphi, _domega])

    def rk44(self, action: np.ndarray):
        [self.a_linear, self.a_angular] = action[:]
        xx = np.array([self.pos[0], self.pos[1], self.vel, self.phi, self.omega])
        K1 = self.dt * self.ode(xx)
        K2 = self.dt * self.ode(xx + K1 / 2)
        K3 = self.dt * self.ode(xx + K2 / 2)
        K4 = self.dt * self.ode(xx + K3)
        xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        [self.pos[0], self.pos[1], self.vel, self.phi, self.omega] = xx[:]
        if self.vel < 0.:
            self.vel = 0.
        self.time += self.dt

        if self.phi > np.pi:
            self.phi -= 2 * np.pi
        if self.phi < -np.pi:
            self.phi += 2 * np.pi

        self.error = self.get_e()
        self.e_phi = self.get_e_phi()

    def get_e(self):
        # forward 位置误差不区分正负
        return np.linalg.norm(self.target - self.pos)

    def get_e_phi(self):
        return cal_vector_rad_oriented([np.cos(self.phi), np.sin(self.phi)], self.target - self.pos)

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

    def load_map_from_datasets(self, datasets):
        self.obs.clear()
        for dataset in datasets:
            pass

    def reset(self, random: bool = True):
        if random:
            self.obs, self.obsNum, self.init_pos, self.init_target = self.generate_circle_obs_training(xMax=self.map_size[0],
                                                                                                       yMax=self.map_size[1],
                                                                                                       safety_dis_obs=4 * self.r_vehicle,
                                                                                                       safety_dis_st=4 * self.r_vehicle,
                                                                                                       rMin=0.2,
                                                                                                       rMax=0.5,
                                                                                                       obsNum=10,
                                                                                                       S=None,
                                                                                                       T=None)
            self.init_phi = np.random.uniform(-np.pi, np.pi)
        self.pos = self.init_pos.copy()
        self.vel = 0.
        self.phi = self.init_phi
        self.omega = 0.
        self.target = self.init_target.copy()
        self.error = self.get_e()
        self.e_phi = self.get_e_phi()
        self.time = 0.
        self.a_linear = self.a_angular = 0.

        self.current_state = self.get_state()
        self.next_state = self.current_state.copy()
        self.current_action = np.array([self.a_linear, self.a_angular])
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功

        self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
        self.draw_init_image()
