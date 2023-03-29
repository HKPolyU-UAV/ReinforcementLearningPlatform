import numpy as np

from common.common_func import *
from environment.envs import *
import math

'''保存的参数
                 m: float = 1.5,
                 g: float = 9.8,
                 Jxx: float = 1.75e-2,
                 Jyy: float = 1.75e-2,
                 Jzz: float = 3.18e-2,
                 d: float = 0.45,
                 CT: float = 1.11e-5,
                 CM: float = 1.49e-7,
                 J0: float = 9.90e-5,
保存的参数'''


class UAV:
    def __init__(self,
                 m: float = 0.8,
                 g: float = 9.8,
                 Jxx: float = 4.212e-3,
                 Jyy: float = 4.212e-3,
                 Jzz: float = 8.255e-3,
                 d: float = 0.12,
                 CT: float = 2.168e-6,
                 CM: float = 2.136e-8,
                 J0: float = 1.01e-5,
                 pos0=None,
                 vel0=None,
                 angle0=None,
                 omega0_inertial=None,
                 omega0_body=None,
                 ):
        if omega0_body is None:
            omega0_body = np.array([0, 0, 0])
        if omega0_inertial is None:
            omega0_inertial = np.array([0, 0, 0])
        if angle0 is None:
            angle0 = np.array([0, 0, 0])
        if vel0 is None:
            vel0 = np.array([0, 0, 0])
        if pos0 is None:
            pos0 = np.array([0, 0, 0])
        '''physical parameters'''
        self.m = m      # 无人机质量
        self.g = g      # 重力加速度
        self.Jxx = Jxx  # X方向转动惯量
        self.Jyy = Jyy  # Y方向转动惯量
        self.Jzz = Jzz  # Z方向转动惯量
        self.d = d      # 机臂长度 'X'构型
        self.CT = CT    # 螺旋桨升力系数
        self.CM = CM    # 螺旋桨力矩系数
        self.J0 = J0    # 电机和螺旋桨的转动惯量

        self.pos = pos0                             # 无人机在世界坐标系下的初始位置
        self.vel = vel0                             # 无人机在世界坐标系下的初始速度
        self.angle = angle0                         # 无人机在世界坐标系下的初始角度
        self.omega_inertial = omega0_inertial       # 无人机在世界坐标系下的初始角速度
        self.omega_body = omega0_body               # 无人机在机体坐标系下的初始角速度

        self.power_allocation_mat = \
            np.array([[CT, CT, CT, CT],
                      [CT * d / math.sqrt(2), -CT * d / math.sqrt(2), -CT * d / math.sqrt(2), CT * d / math.sqrt(2)],
                      [-CT * d / math.sqrt(2), -CT * d / math.sqrt(2), CT * d / math.sqrt(2), CT * d / math.sqrt(2)],
                      [-CM, CM, -CM, CM]])  # 这个矩阵满秩

        self.dt = 0.01  # 控制频率，100Hz
        self.time = 0.  # 当前时间
        self.tmax = 30  # 每回合最大时间

        self.control_state = np.concatenate((self.pos, self.vel, self.angle, self.omega_inertial))  # 控制系统的状态，不是强化学习的状态

        'state limitation'
        self.pos_min = np.array([-10, -10, -10])
        self.pos_max = np.array([10, 10, 10])
        self.vel_min = np.array([-10, -10, -10])
        self.vel_max = np.array([10, 10, 10])
        self.angle_min = np.array([-deg2rad(80), -deg2rad(80), -deg2rad(180)])
        self.angle_max = np.array([deg2rad(80), deg2rad(80), deg2rad(180)])
        self.dangle_min = np.array([-deg2rad(360 * 3), -deg2rad(360 * 3), -deg2rad(360 * 2)])
        self.dangle_max = np.array([deg2rad(360 * 3), deg2rad(360 * 3), deg2rad(360 * 2)])
        'state limitation'

        'control'
        '''
        这里直接使用力作为输入，范围小，方便学习。
        解算微分方程时，直接利用升力系数将力换算成电机角速度即可
        '''
        self.force = np.array([0, 0, 0, 0])
        self.fmin = 0
        self.fmax = 10
        self.w_rotor = np.sqrt(self.force/self.CT)
        'control'

        '''physical parameters'''
        self.is_terminal = False
        self.terminal_flag = 0

        '''datasave'''
        self.save_pos = np.atleast_2d(self.pos)
        self.save_vel = np.atleast_2d(self.vel)
        self.save_angle = np.atleast_2d(self.angle)
        self.save_omega_inertial = np.atleast_2d(self.omega_inertial)
        self.save_omega_body = np.atleast_2d(self.omega_body)
        self.save_f = np.atleast_2d(self.force)
        self.save_t = np.array([self.time])
        '''datasave'''

    def set_position_limitation2inf(self):
        """
        :func:      将无人机的位置限制设置为infinity，该函数仅在设计位置控制器时应用
        :return:
        """
        self.pos_max = np.array([np.inf, np.inf, np.inf])
        self.pos_min = np.array([-np.inf, -np.inf, -np.inf])

    def is_out(self):
        """
        :return:
        """
        # is_omg_out = (self.dphi > self.dphimax + 1e-1) or (self.dphi < self.dphimin - 1e-1) or \
        #              (self.dtheta > self.dthetamax + 1e-1) or (self.dtheta < self.dthetamin - 1e-1) or \
        #              (self.dpsi > self.dpsimax + 1e-1) or (self.dpsi < self.dpsimin - 1e-1)
        # if is_omg_out:
        #     print('Omega out...')
        #     return True

        if np.sum(self.angle > self.angle_max + deg2rad(2)) + np.sum(self.angle < self.angle_min - deg2rad(2)) > 0:
            print('Attitude out...')
            return True

        # is_vel_out = (self.vx > self.vxmax + 1e-1) or (self.vx < self.vxmin - 1e-1) or \
        #              (self.vy > self.vymax + 1e-1) or (self.vy < self.vymin - 1e-1) or \
        #              (self.vz > self.vzmax + 1e-1) or (self.vz < self.vzmin - 1e-1)
        # if is_vel_out:
        #     print('Velocity out...')
        #     return True

        if np.sum(self.pos > self.pos_max + 1e-2) + np.sum(self.pos < self.pos_min - 1e-2) > 0:
            print('Position out...')
            return True

        return False

    def is_episode_Terminal(self):
        self.terminal_flag = 0
        if self.time > self.tmax:
            print('Time out...')
            self.terminal_flag = 2
            self.is_terminal = True
            return True

        if self.is_out():
            print('Out...')
            self.terminal_flag = 1
            self.is_terminal = True
            return True

        self.is_terminal = False
        return False

    def f2omega(self):
        self.w_rotor = np.sqrt(self.force/self.CT)

    def ode(self, xx: np.ndarray):
        """
        :param xx:      状态
        :return:        状态的导数
        """
        '''
        在微分方程里面的状态 X = [x y z vx vy vz phi theta psi p q r] 一共12个
        定义惯性系到机体系的旋转矩阵
        R_i_b1 = np.array([[math.cos(self.psi), math.sin(self.psi), 0],
                           [-math.sin(self.psi), math.cos(self.psi), 0],
                           [0, 0, 1]])  # 从惯性系到b1系，旋转偏航角psi
        R_b1_b2 = np.array([[math.cos(self.theta), 0, -math.sin(self.theta)],
                            [0, 1, 0],
                            [math.sin(self.theta), 0, math.cos(self.theta)]])  # 从b1系到b2系，旋转俯仰角theta
        R_b2_b = np.array([[1, 0, 0],
                           [0, math.cos(self.phi), math.sin(self.phi)],
                           [0, -math.sin(self.phi), math.cos(self.phi)]])  # 从b2系到b系，旋转滚转角phi
        R_i_b = np.matmul(R_b2_b, np.matmul(R_b1_b2, R_i_b1))  # 从惯性系到机体系的转换矩阵
        R_b_i = R_i_b.T  # 从机体系到惯性系的转换矩阵
        e3_i = np.array([0, 0, 1])  # 惯性系下的Z轴基向量
        # dx = v
        # dv = g*e3_i + f/m*
        '''
        [_x, _y, _z, _vx, _vy, _vz, _phi, _theta, _psi, _p, _q, _r] = xx[0:12]
        self.f2omega()  # 根据力，计算出四个电机的转速
        _f = np.sum(self.force)     # 总推力
        square_w = self.w_rotor ** 2
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''
        dp = (self.CT * self.d / math.sqrt(2) * np.dot(square_w, [1, -1, -1, 1]) +
              (self.Jyy - self.Jzz) * _q * _r -
              self.J0 * _q * (self.w_rotor[0] - self.w_rotor[1] + self.w_rotor[2] - self.w_rotor[3])) / self.Jxx
        dq = (self.CT * self.d / math.sqrt(2) * np.dot(square_w, [-1, -1, 1, 1]) +
              (self.Jzz - self.Jxx) * _p * _r -
              self.J0 * _p * (-self.w_rotor[0] + self.w_rotor[1] - self.w_rotor[2] + self.w_rotor[3])) / self.Jyy
        dr = (self.CM * np.dot(square_w, [-1, 1, -1, 1]) + (self.Jxx - self.Jyy) * _p * _q) / self.Jzz
        '''1. 无人机绕机体系旋转的角速度 p q r 的微分方程'''

        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''
        _R_pqr2diner = np.array([[1, math.tan(_theta) * math.sin(_phi), math.tan(_theta) * math.cos(_phi)],
                                 [0, math.cos(_phi), -math.sin(_phi)],
                                 [0, math.sin(_phi) / math.cos(_theta), math.cos(_phi) / math.cos(_theta)]])
        [dphi, dtheta, dpsi] = np.dot(_R_pqr2diner, [_p, _q, _r]).tolist()
        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''

        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''
        [dx, dy, dz] = [_vx, _vy, _vz]
        dvx = _f / self.m * (math.cos(_psi) * math.sin(_theta) * math.cos(_phi) + math.sin(_psi) * math.sin(_phi))
        dvy = _f / self.m * (math.sin(_psi) * math.sin(_theta) * math.cos(_phi) - math.cos(_psi) * math.sin(_phi))
        dvz = -self.g + _f / self.m * math.cos(_phi) * math.cos(_theta)
        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''
        # dvx, dvy, dvz = 0, 0, 0     # 将速度强行限制
        return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])

    def rk44(self, action: np.ndarray):
        self.force = action
        h = self.dt / 1  # RK-44 解算步长
        tt = self.time + self.dt
        while self.time < tt:
            xx_old = np.concatenate((self.pos, self.vel, self.angle, self.omega_body))
            K1 = h * self.ode(xx_old)
            K2 = h * self.ode(xx_old + K1 / 2)
            K3 = h * self.ode(xx_old + K2 / 2)
            K4 = h * self.ode(xx_old + K3)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            xx_temp = xx_new.copy()
            self.pos = xx_temp[0:3]
            self.vel = xx_temp[3:6]
            self.angle = xx_temp[6:9]
            self.omega_body = xx_temp[9:12]
            self.time += h
        R_pqr2diner = np.array([[1, math.tan(self.angle[1]) * math.sin(self.angle[0]), math.tan(self.angle[1]) * math.cos(self.angle[0])],
                                [0, math.cos(self.angle[0]), -math.sin(self.angle[0])],
                                [0, math.sin(self.angle[0]) / math.cos(self.angle[1]), math.cos(self.angle[0]) / math.cos(self.angle[1])]])
        self.omega_inertial = np.dot(R_pqr2diner, self.omega_body)
        if self.angle[2] > np.pi:       # 如果角度超过180度
            self.angle[2] -= 2 * np.pi
        if self.angle[2] < -np.pi:      # 如果角度小于-180度
            self.angle[2] += 2 * np.pi

    def saveData(self, is2file=False, filename='uav.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'x:': self.save_pos[:, 0],
                'y': self.save_pos[:, 1],
                'z': self.save_pos[:, 2],
                'vx': self.save_vel[:, 0],
                'vy': self.save_vel[:, 1],
                'xz': self.save_vel[:, 2],
                'phi': self.save_angle[:, 0],
                'theta': self.save_angle[:, 1],
                'psi': self.save_angle[:, 2],
                'dphi': self.save_omega_inertial[:, 0],
                'dtheta': self.save_omega_inertial[:, 1],
                'dpsi': self.save_omega_inertial[:, 2],
                'p': self.save_omega_body[:, 0],
                'q': self.save_omega_body[:, 1],
                'r': self.save_omega_body[:, 2],
                'f1': self.save_f[:, 0],
                'f2': self.save_f[:, 1],
                'f3': self.save_f[:, 2],
                'f4': self.save_f[:, 3],
                'time': self.save_t
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_pos = np.vstack((self.save_pos, self.pos))
            self.save_vel = np.vstack((self.save_vel, self.vel))
            self.save_angle = np.vstack((self.save_angle, self.angle))
            self.save_omega_inertial = np.vstack((self.save_omega_inertial, self.omega_inertial))
            self.save_omega_body = np.vstack((self.save_omega_body, self.omega_body))
            self.save_f = np.vstack((self.save_f, self.force))
            self.save_t = np.array((self.save_t, self.time))

    def show_uav_linear_state(self, with_time=True):
        np.set_printoptions(precision=3, suppress=True)
        if with_time:
            s = 'Time: %.3fs  Pos: {}m  Vel: {}m/s'.format(self.pos, self.vel) % self.time
        else:
            s = 'Pos: {}m  Vel: {}m/s'.format(self.pos, self.vel)
        print(s)

    def show_uav_angular_state(self, with_time=True):
        np.set_printoptions(precision=3, suppress=True)
        if with_time:
            s = 'Time: %.3fs  Ang: {}rad  dAng: {}rad/s  Omg: {}rad/s'.format(self.angle, self.omega_inertial, self.omega_body) % self.time
        else:
            s = 'Ang: {}rad  dAng: {}rad/s  Omg: {}rad/s'.format(self.angle, self.omega_inertial, self.omega_body)
        print(s)

    def reset(self, pos0: np.array, vel0: np.array, angle0: np.array, omega0_inertial: np.array, omega0_body: np.array):
        self.pos = pos0
        self.vel = vel0
        self.angle = angle0
        self.omega_inertial = omega0_inertial
        self.omega_body = omega0_body
        self.time = 0
        self.control_state = np.concatenate((self.pos, self.vel, self.angle, self.omega_inertial))  # 控制系统的状态，不是强化学习的状态
        self.force = np.array([0, 0, 0, 0])
        self.w_rotor = np.sqrt(self.force / self.CT)
        self.is_terminal = False
        self.terminal_flag = 0

        '''datasave'''
        self.save_pos = np.atleast_2d(self.pos)
        self.save_vel = np.atleast_2d(self.vel)
        self.save_angle = np.atleast_2d(self.angle)
        self.save_omega_inertial = np.atleast_2d(self.omega_inertial)
        self.save_omega_body = np.atleast_2d(self.omega_body)
        self.save_f = np.atleast_2d(self.force)
        self.save_t = np.array([self.time])
        '''datasave'''
