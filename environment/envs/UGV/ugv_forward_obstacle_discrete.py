import cv2 as cv
from common.common_func import *
from environment.envs import *
from environment.envs.UGV.ugv_forward_discrete import UGV_Forward_Discrete as UGV

class UGV_Forward_Obstacle_Discrete(UGV):
    def __init__(self,
                 initPhi: float,
                 save_cfg: bool,
                 x_size: float,
                 y_size: float,
                 start: list,
                 terminal: list,
                 dataBasePath: str = './pathplanning/5X5-50X50-DataBase-AllCircle/'):
        """
        :param initPhi:             initial heading angle
        :param save_cfg:            save to model file or not
        :param x_size:              map size X
        :param y_size:              map size Y
        :param start:               start position
        :param terminal:            terminal position
        :param dataBasePath:        path of the database
        """
        super(UGV_Forward_Obstacle_Discrete, self).__init__(initPhi, save_cfg, x_size, y_size, start, terminal)
        '''physical parameters'''
        self.dt = 0.1       # 10Hz
        self.timeMax = 15.0
        self.staticGain = 2
        self.miss = 1.0 * self.rBody
        # 基本参数都继承了UGV，以上几个是重写的

        self.laserDis = 2.0  # 雷达探测半径
        self.laserBlind = 0.0  # 雷达盲区
        self.laserRange = deg2rad(90)  # 左右各90度，一共180度
        self.laserStep = deg2rad(5)
        self.laserState = int(2 * self.laserRange / self.laserStep) + 1  # 雷达的线数
        self.visualLaser = [[0, 0] for _ in range(self.laserState)]  # 探测点坐标
        self.visualFlag = [0 for _ in range(self.laserState)]  # 探测点类型
        '''physical parameters'''

        '''map database'''
        self.database = self.load_database(dataBasePath)  # 地图数据库
        self.numData = len(self.database)  # 数据库里面的数据数量
        '''map database'''

        '''rl_base'''
        self.state_dim = 8 + self.laserState
        '''[ex, ey, x, y, phi, dx, dy, dphi, laser]'''
        self.state_num = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None, None, None, None, None, None, None, None]
        self.state_space = [None, None, None, None, None, None, None, None]
        self.isStateContinuous = [True, True, True, True, True, True, True, True]
        self.state_range = [
            [-self.staticGain, self.staticGain],
            [-self.staticGain, self.staticGain],
            [0, self.staticGain],
            [0, self.staticGain],
            [-math.pi, math.pi],
            [-self.r * self.wMax, self.r * self.wMax],
            [-self.r * self.wMax, self.r * self.wMax],
            [-self.r / self.L * self.wMax, self.r / self.L * self.wMax]
        ]
        for _ in range(self.laserState):
            self.state_num.append(2)
            self.state_step.append(None)
            self.state_space.append(None)
            self.isStateContinuous.append(False)
            # self.state_range.append([-dis_two_points([self.x_size, self.y_size], [0, 0]), dis_two_points([self.x_size, self.y_size], [0, 0])])
            self.state_range.append([self.laserBlind, self.laserDis])

        self.initial_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 2
        self.action_step = [None, None]
        self.action_range = [[0, self.wMax], [0, self.wMax]]  # only forward
        self.action_num = [int(self.wMax / self.wStep) + 1, int(self.wMax / self.wStep) + 1]
        self.action_space = [[i * self.wStep for i in range(self.action_num[0])], [i * self.wStep for i in range(self.action_num[1])]]
        self.isActionContinuous = [False, False]
        self.initial_action = [0.0, 0.0]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞
        self.trajectory = [self.start]
        '''rl_base'''

        '''visualization_opencv'''
        self.show_dynamic_imagewithobs(isWait=False)
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
            self.saveModel2XML2()

    def draw_fake_laser(self):
        index = 0
        for item in self.visualLaser:
            if self.visualFlag[index] == 0:
                cv.circle(self.image, self.dis2pixel(item), self.length2pixel(0.05), Color().Purple, -1)        # 啥也没有
            elif self.visualFlag[index] == 1:
                cv.circle(self.image, self.dis2pixel(item), self.length2pixel(0.05), Color().LightPink, -1)     # 有东西
            else:
                cv.circle(self.image, self.dis2pixel(item), self.length2pixel(0.05), Color().Red, -1)           # 盲区
            index += 1

    def draw_trajectory(self):
        p1 = self.trajectory[0]
        for p2 in self.trajectory[1:]:
            cv.line(self.image, self.dis2pixel(p1), self.dis2pixel(p2), Color().Green, 2)
            p1 = p2.copy()

    def map_draw_inner_boundary(self):
        cv.rectangle(self.image, self.dis2pixel([0.5, 0.5]), self.dis2pixel([self.x_size - 0.5, self.y_size - 0.5]), Color().Black, 1)

    def show_dynamic_imagewithobs(self, isWait=False):
        self.image = self.image_temp.copy()
        self.map_draw_inner_boundary()
        self.map_draw_obs()
        self.map_draw_photo_frame()
        self.map_draw_boundary()
        self.draw_car()
        self.draw_trajectory()
        self.draw_fake_laser()
        self.draw_terminal()
        cv.putText(self.image, str(round(self.time, 3)), (0, 15), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 1)
        cv.imshow(self.name4image, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)
        self.save = self.image.copy()

    def is_Terminal(self, param=None):
        self.terminal_flag = 0
        if self.time > self.timeMax:
            print('...time out...')
            self.terminal_flag = 2
            return True
        if self.collision_check():
            # print('...collision...')
            self.terminal_flag = 4
            return True
        # if self.delta_phi_absolute > 6 * math.pi + deg2rad(0) and dis_two_points([self.x, self.y], [self.initX, self.initY]) <= 1.0:
        # if self.delta_phi_absolute > 6 * math.pi + deg2rad(0):
        #     print('...转的角度太大了...')
        #     self.terminal_flag = 1
        #     return True
        if dis_two_points([self.x, self.y], self.terminal) <= self.miss:
            print('...success...')
            self.terminal_flag = 3
            return True
        if self.is_out():
            # print('...out...')
            self.terminal_flag = 5
            return True
        return False

    def get_fake_laser(self):
        laser = []
        detectPhi = np.linspace(self.phi - self.laserRange, self.phi + self.laserRange, self.laserState)  # 所有的角度
        count = 0

        '''如果车本身在障碍物里面'''
        if self.collision_check():
            for i in range(self.laserState):
                laser.append(self.laserBlind)
                self.visualLaser[i] = [self.x, self.y]
                self.visualFlag[i] = 1
            return laser
        '''如果车本身在障碍物里面'''

        start = [self.x, self.y]
        '''1. 提前求出起点与障碍物中心距离，然后将距离排序'''
        ref_dis = []
        for _obs in self.obs:
            ref_dis.append(dis_two_points([self.x, self.y], _obs[2]))
        ref_sort = np.argsort(ref_dis)  # 排序的障碍物，距离从小到达，越小的说明离机器人越近
        '''1. 提前求出起点与障碍物中心距离，然后将距离排序'''
        for phi in detectPhi:
            if phi > math.pi:
                phi -= 2 * math.pi
            if phi < -math.pi:
                phi += 2 * math.pi
            m = np.tan(phi)  # 斜率
            b = self.y - m * self.x  # 截距
            '''2. 确定当前机器人与四个角点的连接'''
            theta1 = cal_vector_rad([1, 0], [self.x_size - self.x, self.y_size - self.y])  # 左上
            theta2 = cal_vector_rad([1, 0], [0 - self.x, self.y_size - self.y])  # 右上
            theta3 = -cal_vector_rad([1, 0], [0 - self.x, 0 - self.y])  # 右下
            theta4 = -cal_vector_rad([1, 0], [self.x_size - self.x, 0 - self.y])  # 左下
            '''2. 确定当前机器人与四个角点的连接'''
            '''3. 找到终点'''
            cosTheta = math.fabs(m) / math.sqrt(1 + m ** 2)
            sinTheta = 1 / math.sqrt(1 + m ** 2)
            if theta4 < phi <= theta1:
                terminal = [self.x_size, m * self.x_size + b]
                tx = self.x + self.laserDis / math.sqrt(1 + m ** 2)
                if tx < self.x_size:
                    terminal = [tx, self.y + cosTheta * self.laserDis] if m >= 0 else [tx, self.y - cosTheta * self.laserDis]
            elif theta1 < phi <= theta2:
                terminal = [(self.y_size - b) / m, self.y_size] if math.fabs(m) < 1e8 else [self.x, self.y_size]
                ty = self.y + math.fabs(m) * self.laserDis / math.sqrt(1 + m ** 2)
                if ty < self.y_size:
                    terminal = [self.x + self.laserDis * sinTheta, ty] if m >= 0 else [self.x - self.laserDis * sinTheta, ty]
            elif theta3 < phi <= theta4:
                terminal = [-b / m, 0] if math.fabs(m) < 1e8 else [self.x, 0]
                ty = self.y - math.fabs(m) * self.laserDis / math.sqrt(1 + m ** 2)
                if ty > 0:
                    terminal = [self.x - self.laserDis * sinTheta, ty] if m >= 0 else [self.x + self.laserDis * sinTheta, ty]
            else:
                terminal = [0, b]
                tx = self.x - self.laserDis / math.sqrt(1 + m ** 2)
                if tx > 0:
                    terminal = [tx, self.y - cosTheta * self.laserDis] if m >= 0 else [tx, self.y + cosTheta * self.laserDis]
            '''3. 找到终点'''
            '''4. 开始找探测点'''
            find = False
            for index in ref_sort:
                _obs = self.obs[index]
                x0 = _obs[2][0]
                y0 = _obs[2][1]
                r0 = _obs[1][0]
                if ref_dis[index] > self.laserDis + r0:
                    continue  # 如果障碍物本身超出可探测范围，那么肯定不用考虑
                if np.fabs(m * x0 - y0 + b) / np.sqrt(1 + m ** 2) > r0:
                    continue  # 如果圆心到线段所在直线的距离大于圆的半径，那么肯定不用考虑
                if cal_vector_rad([terminal[0] - start[0], terminal[1] - start[1]], [x0 - start[0], y0 - start[1]]) > math.pi / 2:
                    continue  # 如果圆心的位置在探测线段的后方，那么肯定是不需要考虑
                '''能执行到这，就说明存在一个园，使得线段所在的射线满足条件，只需要计算点是否在线段上即可'''
                # 垂足坐标
                foot_x = (x0 + m * y0 - m * b) / (m ** 2 + 1)
                foot_y = (m * x0 + m ** 2 * y0 + b) / (m ** 2 + 1)
                r_dis = dis_two_points([foot_x, foot_y], [x0, y0])
                # dis_slide = math.sqrt(r0 ** 2 - r_dis ** 2)     # 垂足到交点滑动距离
                crossPtx = foot_x - np.sign(terminal[0] - start[0]) * math.sqrt(r0 ** 2 - r_dis ** 2) / math.sqrt(m ** 2 + 1)
                if min(start[0], terminal[0]) <= crossPtx <= max(start[0], terminal[0]):
                    find = True
                    dis = math.fabs(crossPtx - start[0]) * math.sqrt(m ** 2 + 1)
                    if dis < self.laserBlind:  # too close
                        laser.append(self.laserBlind)
                        newX = start[0] + self.laserBlind / math.sqrt(m ** 2 + 1) * np.sign(terminal[0] - start[0])
                        self.visualLaser[count] = [newX, m * newX + b]
                        self.visualFlag[count] = 2
                    else:
                        laser.append(dis)
                        self.visualLaser[count] = [crossPtx, m * crossPtx + b]
                        self.visualFlag[count] = 1
                    break
            '''4. 开始找探测点'''
            if not find:  # 点一定是终点，但是属性不一定
                dis = dis_two_points(start, terminal)
                # laser.append(self.laserDis)
                # self.visualLaser[count] = terminal.copy()
                # self.visualFlag[count] = 0                  # 只要没有，就给2.0
                if dis > self.laserDis:         # 如果起始点与终点的距离大于探测半径，那么就直接给探测半径，相当于空场地
                    laser.append(self.laserDis)
                    self.visualLaser[count] = terminal.copy()
                    self.visualFlag[count] = 0
                elif self.laserBlind < dis <= self.laserDis:        # 如果起始点与终点的距离小于探测半径，那么直接给距离，说明探测到场地边界
                    laser.append(dis)
                    self.visualLaser[count] = terminal.copy()
                    self.visualFlag[count] = 0
                else:       # 进入雷达盲区，0m
                    laser.append(self.laserBlind)
                    self.visualLaser[count] = terminal.copy()
                    self.visualFlag[count] = 2
            count += 1
        return laser

    def collision_check(self):
        # 假设所有的障碍物都是圆
        for _obs in self.obs:
            if dis_two_points([self.x, self.y], _obs[2]) < _obs[1][0] + self.rBody:
                return True
        return False

    def get_reward(self, param=None):
        cex = self.current_state[0] * self.x_size / self.staticGain
        cey = self.current_state[1] * self.y_size / self.staticGain
        nex = self.next_state[0] * self.x_size / self.staticGain
        ney = self.next_state[1] * self.y_size / self.staticGain
        currentError = math.sqrt(cex ** 2 + cey ** 2)
        nextError = math.sqrt(nex ** 2 + ney ** 2)

        gain = 2.0 * math.sqrt(self.dx ** 2 + self.dy ** 2)

        r1 = -1  # 常值误差，每运行一步，就 -1

        if currentError > nextError + 1e-3:
            r2 = gain * 5
        elif 1e-3 + currentError < nextError:
            r2 = -5
        else:
            r2 = 0

        currentTheta = cal_vector_rad([cex, cey], [math.cos(self.current_state[4]), math.sin(self.current_state[4])])
        nextTheta = cal_vector_rad([nex, ney], [math.cos(self.next_state[4]), math.sin(self.next_state[4])])
        # print(currentTheta, nextTheta)
        if currentTheta > nextTheta + 1e-2:
            r3 = gain * 2
        elif 1e-2 + currentTheta < nextTheta:
            r3 = -2
        else:
            r3 = 0
        # r3 = 0          # 不给角度惩罚

        '''4. 其他'''
        if self.terminal_flag == 3:  # 成功了
            r4 = 100
        elif self.terminal_flag == 1:  # 转的角度太大了
            r4 = 0
        elif self.terminal_flag == 4:  # 碰撞障碍物
            r4 = -10
        else:
            r4 = 0
        '''4. 其他'''

        self.reward = r1 + r2 + r3 + r4

    def step_update(self, action: list):
        self.wLeft = max(min(action[0], self.wMax), 0)
        self.wRight = max(min(action[1], self.wMax), 0)
        self.current_action = action.copy()
        self.current_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
        # self.state_normalization(self.current_state, gain=self.staticGain, index0=0, index1=3)
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
        '''RK-44'''
        '''动力学系统状态更新'''
        [self.x, self.y, self.phi] = list(state)
        self.dx = self.r / 2 * (self.wLeft + self.wRight) * math.cos(self.phi)
        self.dy = self.r / 2 * (self.wLeft + self.wRight) * math.sin(self.phi)
        self.dphi = self.r / self.rBody * (self.wRight - self.wLeft)
        self.time += self.dt
        '''动力学系统状态更新'''
        self.delta_phi_absolute += math.fabs(self.phi - self.current_state[4])
        '''角度处理'''
        if self.phi > math.pi:
            self.phi -= 2 * math.pi
        if self.phi < -math.pi:
            self.phi += 2 * math.pi
        '''角度处理'''
        self.is_terminal = self.is_Terminal()
        self.next_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                           (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                           self.x / self.x_size * self.staticGain,
                           self.y / self.y_size * self.staticGain,
                           self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
        '''出界处理'''
        if self.x + self.rBody > self.x_size:  # Xout
            self.x = self.x_size - self.rBody
            self.dx = 0
        if self.x - self.rBody < 0:
            self.x = self.rBody
            self.dx = 0
        if self.y + self.rBody > self.y_size:  # Yout
            self.y = self.y_size - self.rBody
            self.dy = 0
        if self.y - self.rBody < 0:
            self.y = self.rBody
            self.dy = 0
        '''出界处理'''

        '''处理轨迹绘制'''
        self.trajectory.append([self.x, self.y])
        '''处理轨迹绘制'''

        # self.state_normalization(self.next_state, gain=self.staticGain, index0=0, index1=3)
        self.get_reward()
        self.saveData()

        return self.current_state, action, self.reward, self.next_state, self.is_terminal

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
        self.delta_phi_absolute = 0.
        self.trajectory = [self.start]
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

    def reset_random(self, uniform=False):
        """
        :return:
        """
        '''physical parameters'''
        self.set_start([random.uniform(self.rBody, self.x_size - self.rBody), random.uniform(self.rBody, self.y_size - self.rBody)])
        self.set_terminal([random.uniform(self.rBody, self.x_size - self.rBody), random.uniform(self.rBody, self.y_size - self.rBody)])
        self.set_random_obstacles(3)
        self.x = self.start[0]  # X
        self.y = self.start[1]  # Y
        self.initX = self.start[0]
        self.initY = self.start[1]

        phi0 = cal_vector_rad([self.terminal[0] - self.x, self.terminal[1] - self.y], [1, 0])
        phi0 = phi0 if self.y <= self.terminal[1] else -phi0
        # print(rad2deg(phi0))
        self.phi = random.uniform(phi0 - deg2rad(45), phi0 + deg2rad(45))  # 将初始化的角度放在初始对准目标的90度范围内
        '''角度处理'''
        if self.phi > math.pi:
            self.phi -= 2 * math.pi
        if self.phi < -math.pi:
            self.phi += 2 * math.pi
        '''角度处理'''
        self.initPhi = self.phi

        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.
        self.time = 0.  # time
        self.delta_phi_absolute = 0.
        self.trajectory = [self.start]
        '''physical parameters'''

        '''RL_BASE'''
        self.initial_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
        # self.state_normalization(self.initial_state, gain=self.staticGain, index0=0, index1=3)
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

    def reset_random_with_database(self):
        num = random.randint(0, self.numData - 1)
        data = self.database[num]
        '''physical parameters and map'''
        self.start = data[0]
        self.terminal = data[1]
        self.obs = data[3]
        self.x = self.start[0]  # X
        self.y = self.start[1]  # Y
        self.initX = self.start[0]
        self.initY = self.start[1]
        phi0 = cal_vector_rad([self.terminal[0] - self.x, self.terminal[1] - self.y], [1, 0])
        phi0 = phi0 if self.y <= self.terminal[1] else -phi0
        # print(rad2deg(phi0))
        self.phi = random.uniform(phi0 - deg2rad(45), phi0 + deg2rad(45))  # 将初始化的角度放在初始对准目标的90度范围内
        '''角度处理'''
        if self.phi > math.pi:
            self.phi -= 2 * math.pi
        if self.phi < -math.pi:
            self.phi += 2 * math.pi
        '''角度处理'''
        self.initPhi = self.phi
        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.
        self.time = 0.  # time
        self.delta_phi_absolute = 0.
        self.trajectory = [self.start]
        '''physical parameters and map'''

        '''RL_BASE'''
        self.initial_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
        # self.state_normalization(self.initial_state, gain=self.staticGain, index0=0, index1=3)
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

    def saveModel2XML2(self, filename='UGV_Forward_Obstacle_Discrete.xml', filepath='../config/'):
        rootMsg = {
            'name': 'UGV_Forward_Obstacle_Continuous',
            'author': 'Yefeng YANG',
            'date': '2022.01.20',
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
            'wStep': self.wStep,
            'wMax': self.wMax,
            'r': self.r,
            'l_wheel': self.l_wheel,
            'rBody': self.rBody,
            'L': self.L,
            'dt': self.dt,
            'time': self.time,
            'x_size': self.x_size,
            'y_size': self.y_size,
            'miss': self.miss,
            'laserDis': self.laserDis,
            'laserBlind': self.laserBlind,
            'laserRange': self.laserRange,
            'laserStep': self.laserStep,
            'laserState': self.laserState,
            'visualLaser': self.visualLaser,
            'visualFlag': self.visualFlag,
            'numData': self.numData
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
                                 nodename='Tips',
                                 nodemsg={'comment': 'Some attributes are directly inherited from UGV_Forward_Continuous'})
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

    def saveData(self, is2file=False, filename='UGV_Forward_Continuous_Discrete.csv', filepath=''):
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

    def load_database(self, path):
        DataBase = []
        names = os.listdir(path)
        for name in names:
            print('Start Loading' + name)
            DataBase.append(self.map_load_continuous_database(path + name))
            print('Finish Loading' + name)
        return self.merge_database(DataBase)

