import numpy as np
import cv2 as cv
from uav import UAV, uav_param
from collector import data_collector
from FNTSMC import fntsmc_att, fntsmc_pos, fntsmc_param
from ref_cmd import *
from environment.color import Color


class uav_pos_ctrl(UAV):
    def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param, pos_ctrl_param: fntsmc_param):
        super(uav_pos_ctrl, self).__init__(UAV_param)
        self.att_ctrl = fntsmc_att(att_ctrl_param)
        self.pos_ctrl = fntsmc_pos(pos_ctrl_param)

        self.collector = data_collector(round(self.time_max / self.dt))
        self.collector.reset(round(self.time_max / self.dt))

        self.pos_ref = np.zeros(3)
        self.dot_pos_ref = np.zeros(3)
        self.att_ref = np.zeros(3)
        self.att_ref_old = np.zeros(3)
        self.dot_att_ref = np.zeros(3)

        self.dot_att_ref_limit = 60. * np.pi / 180. * np.ones(3)  # 最大角速度不能超过 60 度 / 秒

        self.obs = np.zeros(3)  # output of the observer
        self.dis = np.zeros(3)  # external disturbance, known by me, but not the controller

        '''参考轨迹记录'''
        self.ref_amplitude = None
        self.ref_period = None
        self.ref_bias_a = None
        self.ref_bias_phase = None
        self.trajectory = None
        '''参考轨迹记录'''

        '''opencv visualization for position control'''
        self.width = 1200
        self.height = 400
        self.x_offset = 40
        self.y_offset = 40
        self.offset = 20
        self.wp = (self.width - 2 * self.x_offset - 4 * self.offset) / 3
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min
        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        self.pmx_p1 = self.wp / dx
        self.pmy_p1 = (self.height - 2 * self.y_offset) / dy
        self.pmx_p2 = self.wp / dy
        self.pmy_p2 = (self.height - 2 * self.y_offset) / dz
        self.pmx_p3 = self.wp / dz
        self.pmy_p3 = (self.height - 2 * self.y_offset) / dx
        '''opencv visualization for position control'''

    def dis2pixel(self, coord, flag: str, offset):
        if flag == 'xoy':
            x = self.x_offset + (coord[0] - self.x_min) * self.pmx_p1
            y = self.height - self.y_offset - (coord[1] - self.y_min) * self.pmy_p1
            return int(x + offset[0]), int(y + offset[1])
        if flag == 'yoz':
            y = self.x_offset + (coord[1] - self.y_min) * self.pmx_p2
            z = self.height - self.y_offset - (coord[2] - self.z_min) * self.pmy_p2
            return int(y + offset[0]), int(z + offset[1])
        if flag == 'zox':
            z = self.x_offset + (coord[2] - self.z_min) * self.pmx_p3
            x = self.height - self.y_offset - (coord[0] - self.x_min) * self.pmy_p3
            return int(z + offset[0]), int(x + offset[1])
        return offset[0], offset[1]

    def dis2pixel_trajectory_numpy2d(self, traj: np.ndarray, flag: str, offset: list) -> np.ndarray:
        """
        @param traj:        无人机轨迹，N * 3
        @param flag:        xoy yoz zox
        @param offset:      偏移
        @return:
        """
        if flag == 'xoy':
            x = self.x_offset + (traj[:, 0] - self.x_min) * self.pmx_p1 + offset[0]
            y = self.height - self.y_offset - (traj[:, 1] - self.y_min) * self.pmy_p1 + offset[1]
            return np.vstack((x, y)).T
        if flag == 'yoz':
            y = self.x_offset + (traj[:, 1] - self.y_min) * self.pmx_p2 + offset[0]
            z = self.height - self.y_offset - (traj[:, 2] - self.z_min) * self.pmy_p2 + offset[1]
            return np.vstack((y, z)).T
        if flag == 'zox':
            z = self.x_offset + (traj[:, 2] - self.z_min) * self.pmx_p3 + offset[0]
            x = self.height - self.y_offset - (traj[:, 0] - self.x_min) * self.pmy_p3 + offset[1]
            return np.vstack((z, x)).T
        return np.array([])

    def draw_boundary_xoy(self):
        cv.rectangle(self.image,
                     self.dis2pixel([self.x_min, self.y_min, 0], 'xoy', [0, 0]),
                     self.dis2pixel([self.x_max, self.y_max, 0], 'xoy', [0, 0]),
                     Color().Black, 2)

    def draw_boundary_yoz(self):
        pt1 = self.dis2pixel([0, self.y_min, self.z_min],
                             'yoz',
                             [self.wp + 2 * self.offset, 0])
        pt2 = self.dis2pixel([0, self.y_max, self.z_max],
                             'yoz',
                             [self.wp + 2 * self.offset, 0])
        cv.rectangle(self.image, pt1, pt2, Color().Black, 2)

    def draw_boundary_zox(self):
        cv.rectangle(self.image,
                     self.dis2pixel([self.x_min, 0, self.z_min],
                                    'zox',
                                    [2 * self.wp + 4 * self.offset, 0]),
                     self.dis2pixel([self.x_max, 0, self.z_max],
                                    'zox',
                                    [2 * self.wp + 4 * self.offset, 0]),
                     Color().Black, 2)

    def draw_boundary(self):
        self.draw_boundary_xoy()
        self.draw_boundary_yoz()
        self.draw_boundary_zox()

    def draw_label(self):
        pts = [self.dis2pixel([(self.x_min + self.x_max) / 2, self.y_min, 0], 'xoy', [-5, -5]),
               self.dis2pixel([self.x_min, (self.y_min + self.y_max) / 2, 0], 'xoy', [5, 0]),
               self.dis2pixel([0, (self.y_min + self.y_max) / 2, self.z_min], 'yoz', [self.wp + 2 * self.offset - 5, -5]),
               self.dis2pixel([0, self.y_min, (self.z_min + self.z_max) / 2], 'yoz', [self.wp + 2 * self.offset + 5, 0]),
               self.dis2pixel([self.x_min, 0, (self.z_min + self.z_max) / 2], 'zox', [2 * self.wp + 4 * self.offset - 5, -5]),
               self.dis2pixel([(self.x_min + self.x_max) / 2, 0, self.z_min], 'zox', [2 * self.wp + 4 * self.offset + 5, 0]),
               (int(self.width / 2 - 55), 20)]
        labels = ['X', 'Y', 'Y', 'Z', 'Z', 'X', 'Projection']
        for _l, _pt in zip(labels, pts):
            cv.putText(self.image, _l, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

    def draw_region_grid(self, xNum: int, yNum: int, zNum: int):
        _dx = (self.x_max - self.x_min) / xNum
        _dy = (self.y_max - self.y_min) / yNum
        _dz = (self.z_max - self.z_min) / zNum

        '''X'''
        for i in range(yNum - 1):
            pt1 = self.dis2pixel([self.x_min, self.y_min + (i + 1) * _dy, 0.], 'xoy', [0, 0])
            pt2 = self.dis2pixel([self.x_max, self.y_min + (i + 1) * _dy, 0.], 'xoy', [0, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        for i in range(zNum - 1):
            pt1 = self.dis2pixel([self.x_min, 0., self.z_min + (i + 1) * _dz], 'zox', [2 * self.wp + 4 * self.offset, 0])
            pt2 = self.dis2pixel([self.x_max, 0., self.z_min + (i + 1) * _dz], 'zox', [2 * self.wp + 4 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)

        '''Y'''
        for i in range(xNum - 1):
            pt1 = self.dis2pixel([self.x_min + (i + 1) * _dx, self.y_min, 0.], 'xoy', [0, 0])
            pt2 = self.dis2pixel([self.x_min + (i + 1) * _dx, self.y_max, 0.], 'xoy', [0, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        for i in range(zNum - 1):
            pt1 = self.dis2pixel([0., self.y_min, self.z_min + (i + 1) * _dz], 'yoz', [self.wp + 2 * self.offset, 0])
            pt2 = self.dis2pixel([0., self.y_max, self.z_min + (i + 1) * _dz], 'yoz', [self.wp + 2 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)

        '''Z'''
        for i in range(yNum - 1):
            pt1 = self.dis2pixel([0., self.y_min + (i + 1) * _dy, self.z_min], 'yoz', [self.wp + 2 * self.offset, 0])
            pt2 = self.dis2pixel([0., self.y_min + (i + 1) * _dy, self.z_max], 'yoz', [self.wp + 2 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)
        for i in range(xNum - 1):
            pt1 = self.dis2pixel([self.x_min + (i + 1) * _dx, 0., self.z_min], 'zox', [2 * self.wp + 4 * self.offset, 0])
            pt2 = self.dis2pixel([self.x_min + (i + 1) * _dx, 0., self.z_max], 'zox', [2 * self.wp + 4 * self.offset, 0])
            cv.line(self.image, pt1, pt2, Color().Black, 1)

        self.draw_axis(xNum, yNum, zNum)

    def draw_axis(self, xNum: int, yNum: int, zNum: int):
        _dx = (self.x_max - self.x_min) / xNum
        _dy = (self.y_max - self.y_min) / yNum
        _dz = (self.z_max - self.z_min) / zNum

        _x = np.linspace(self.x_min, self.x_max, xNum + 1)
        _y = np.linspace(self.y_min, self.y_max, yNum + 1)
        _z = np.linspace(self.z_min, self.z_max, zNum + 1)

        for __x in _x:
            if np.fabs(round(__x, 2) - int(__x)) < 0.01:
                _s = str(int(__x))
            else:
                _s = str(round(__x, 2))
            _pt = self.dis2pixel([__x, self.y_min, 0], 'xoy', [-20 if __x < 0 else -7, 20])
            _pt2 = self.dis2pixel([__x, 0., self.z_min],
                                  'zox',
                                  [2 * self.wp + 4 * self.offset - 30 if __x < 0 else 2 * self.wp + 4 * self.offset - 15, 5])
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
            cv.putText(self.image, _s, _pt2, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

        for __y in _y:
            if np.fabs(round(__y, 2) - int(__y)) < 0.01:
                _s = str(int(__y))
            else:
                _s = str(round(__y, 2))
            _pt = self.dis2pixel([self.x_min, __y, 0], 'xoy', [-30 if __y < 0 else -15, 7])
            _pt2 = self.dis2pixel([0., __y, self.z_min],
                                  'yoz',
                                  [self.wp + 2 * self.offset - 15 if __y < 0 else self.wp + 2 * self.offset - 5, 20])
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
            cv.putText(self.image, _s, _pt2, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

        for __z in _z:
            if np.fabs(round(__z, 2) - int(__z)) < 0.01:  # 是整数
                _s = str(int(__z))
                _pt = self.dis2pixel([0., self.y_min, __z], 'yoz', [self.wp + 2 * self.offset - 20, 7])
                _pt2 = self.dis2pixel([self.x_min, 0., __z], 'zox', [2 * self.wp + 4 * self.offset - 10, 20])
            else:
                _s = str(round(__z, 2))
                _pt = self.dis2pixel([0., self.y_min, __z], 'yoz', [self.wp + 2 * self.offset - 30, 7])
                _pt2 = self.dis2pixel([self.x_min, 0., __z], 'zox', [2 * self.wp + 4 * self.offset - 15, 20])
            cv.putText(self.image, _s, _pt, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)
            cv.putText(self.image, _s, _pt2, cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

    def draw_3d_points_projection(self, points: np.ndarray, colors: list):
        """
        @param colors:
        @param colors:
        @param points:
        @return:
        """
        '''XOY'''
        xy = self.dis2pixel_trajectory_numpy2d(points, 'xoy', [0, 0])
        _l = xy.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(xy[i][0])), int(round(xy[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)

        '''YOZ'''
        yz = self.dis2pixel_trajectory_numpy2d(points, 'yoz', [self.wp + 2 * self.offset, 0])
        _l = yz.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(yz[i][0])), int(round(yz[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)

        '''ZOX'''
        zx = self.dis2pixel_trajectory_numpy2d(points, 'zox', [2 * self.wp + 4 * self.offset, 0])
        _l = zx.shape[0]  # 一共有多少数据
        for i in range(_l):
            pt1 = (int(round(zx[i][0])), int(round(zx[i][1])))
            cv.circle(self.image, pt1, 5, colors[i], -1)

    def draw_3d_trajectory_projection(self, trajectory: np.ndarray):
        """
        @param trajectory:
        @return:
        """
        '''XOY'''
        xy = self.dis2pixel_trajectory_numpy2d(trajectory, 'xoy', [0, 0])
        _l = xy.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(xy[i][0])), int(round(xy[i][1])))
            pt2 = (int(round(xy[i + 1][0])), int(round(xy[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)

        '''YOZ'''
        yz = self.dis2pixel_trajectory_numpy2d(trajectory, 'yoz', [self.wp + 2 * self.offset, 0])
        _l = yz.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(yz[i][0])), int(round(yz[i][1])))
            pt2 = (int(round(yz[i + 1][0])), int(round(yz[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)

        '''ZOX'''
        zx = self.dis2pixel_trajectory_numpy2d(trajectory, 'zox', [2 * self.wp + 4 * self.offset, 0])
        _l = zx.shape[0]  # 一共有多少数据
        for i in range(_l - 1):
            pt1 = (int(round(zx[i][0])), int(round(zx[i][1])))
            pt2 = (int(round(zx[i + 1][0])), int(round(zx[i + 1][1])))
            cv.line(self.image, pt1, pt2, Color().Blue, 1)

    def draw_time_error(self, pos: np.ndarray, ref: np.ndarray):
        """
        @param pos:
        @param ref:
        @return:
        """
        e = pos - ref
        _str = '[%.2f, %.2f, %.2f]' % (e[0], e[1], e[2])
        cv.putText(self.image, _str, (self.x_offset, self.y_offset - 5), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 2)
        _str = 't = %.2f' % self.time
        cv.putText(self.image, _str, (self.x_offset + 250, self.y_offset - 5), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 2)

    def draw_init_image(self):
        self.draw_boundary()
        self.draw_label()
        self.draw_region_grid(6, 6, 6)
        self.draw_axis(6, 6, 6)
        self.image_copy = self.image.copy()

    def show_image(self, iswait: bool = False):
        if iswait:
            cv.imshow('Projection', self.image)
            cv.waitKey(0)
        else:
            cv.imshow('Projection', self.image)
            cv.waitKey(1)

    def pos_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray):
        """
        @param ref:			x_d y_d z_d
        @param dot_ref:		vx_d vy_d vz_d
        @param dot2_ref:	ax_d ay_d az_d
        @return:			ref_phi ref_theta throttle
        """
        self.pos_ref = ref
        self.dot_pos_ref = dot_ref
        e = self.eta() - ref
        de = self.dot_eta() - dot_ref
        self.pos_ctrl.control_update(self.kt, self.m, self.uav_vel(), e, de, dot2_ref, np.zeros(3))
        phi_d, theta_d, uf = self.uo_2_ref_angle_throttle(limit=[np.pi / 4, np.pi / 4], att_limitation=True)
        return phi_d, theta_d, uf

    def att_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray, att_only: bool = False):
        """
        @param ref:			phi_d theta_d psi_d
        @param dot_ref:		dot_phi_d dot_theta_d dot_psi_d
        @param dot2_ref:
        @param att_only:	为 True 时，dot2_ref 正常输入
                            为 True 时，dot2_ref 为 0
        @return:			Tx Ty Tz
        """
        self.att_ref_old = self.att_ref.copy()
        self.att_ref = ref
        self.dot_att_ref = dot_ref
        if not att_only:
            dot2_ref = np.zeros(3)

        e = self.rho1() - ref
        de = self.dot_rho1() - dot_ref
        sec_order_att_dy = self.second_order_att_dynamics()
        ctrl_mat = self.att_control_matrix()
        self.att_ctrl.control_update(sec_order_att_dy, ctrl_mat, e, de, dot2_ref)
        return self.att_ctrl.control

    def uo_2_ref_angle_throttle(self, limit=None, att_limitation: bool = False):
        """
        @param limit:				期望姿态角限制
        @param att_limitation:		是否使用 limit
        @return:					期望 phi_d theta_d 油门
        """
        ux = self.pos_ctrl.control[0]
        uy = self.pos_ctrl.control[1]
        uz = self.pos_ctrl.control[2]
        uf = (uz + self.g) * self.m / (np.cos(self.phi) * np.cos(self.theta))
        asin_phi_d = min(max((ux * np.sin(self.psi) - uy * np.cos(self.psi)) * self.m / uf, -1), 1)
        phi_d = np.arcsin(asin_phi_d)
        asin_theta_d = min(max((ux * np.cos(self.psi) + uy * np.sin(self.psi)) * self.m / (uf * np.cos(phi_d)), -1), 1)
        theta_d = np.arcsin(asin_theta_d)
        if att_limitation:
            if limit is not None:
                phi_d = max(min(phi_d, limit[0]), -limit[0])
                theta_d = max(min(theta_d, limit[1]), -limit[1])
        return phi_d, theta_d, uf

    def update(self, action: np.ndarray):
        """
        @param action:  油门 + 三个力矩
        @return:
        """
        data_block = {'time': self.time,  # simulation time
                      'control': action,  # actual control command
                      'ref_angle': self.att_ref,  # reference angle
                      'ref_dot_angle': self.dot_att_ref,
                      'ref_pos': self.pos_ref,
                      'ref_vel': self.dot_pos_ref,
                      'd_out': self.dis / self.m,
                      'd_out_obs': self.obs,
                      'state': self.uav_state_call_back(),
                      'dot_angle': self.uav_dot_att()
                      }  # quadrotor state
        self.collector.record(data_block)
        self.rk44(action=action, dis=self.dis, n=1, att_only=False)

    def generate_ref_trajectory(self, _amplitude: np.ndarray, _period: np.ndarray, _bias_a: np.ndarray, _bias_phase: np.ndarray):
        """
        @param _amplitude:
        @param _period:
        @param _bias_a:
        @param _bias_phase:
        @return:
        """
        t = np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1)
        rx = _bias_a[0] + _amplitude[0] * np.sin(2 * np.pi / _period[0] * t + _bias_phase[0])
        ry = _bias_a[1] + _amplitude[1] * np.sin(2 * np.pi / _period[1] * t + _bias_phase[1])
        rz = _bias_a[2] + _amplitude[2] * np.sin(2 * np.pi / _period[2] * t + _bias_phase[2])
        rpsi = _bias_a[3] + _amplitude[3] * np.sin(2 * np.pi / _period[3] * t + _bias_phase[3])
        return np.vstack((rx, ry, rz, rpsi)).T

    def generate_random_trajectory(self, is_random: bool = False, yaw_fixed: bool = True, outer_param:list = None):
        """
        @param is_random:	随机在振幅与周期
        @param yaw_fixed:	偏航角固定
        @return:			None
        """
        center = np.concatenate((np.mean(self.pos_zone, axis=1), [np.mean(self.att_zone[2])]))
        if outer_param is not None:
            A = outer_param[0]
            T = outer_param[1]
            phi0 = outer_param[2]
        else:
            if is_random:
                a = np.random.uniform(low=0, high=1.5)
                A = np.array([a, a, a, 0])
                T = np.random.uniform(low=5, high=10) * np.ones(4)
                phi0 = np.array([np.pi / 2, 0., 0., 0.])
                # A = np.array([
                #     np.random.uniform(low=0., high=self.x_max - center[0]),
                #     np.random.uniform(low=0., high=self.y_max - center[1]),
                #     np.random.uniform(low=0., high=self.z_max - center[2]),
                #     np.random.uniform(low=0, high=self.att_zone[2][1] - center[3])
                # ])
                # T = np.random.uniform(low=5, high=10, size=4)  # 随机生成周期
                # # phi0 = np.random.uniform(low=0, high=np.pi / 2, size=4)
                # phi0 = np.array([np.pi / 2, 0., 0., 0.])
            else:
                A = np.array([1.5, 1.5, 0.3, 0.])
                T = np.array([6., 6., 10, 10])
                phi0 = np.array([np.pi / 2, 0., 0., 0.])

            if yaw_fixed:
                A[3] = 0.
                phi0[3] = 0.

        self.ref_amplitude = A
        self.ref_period = T
        self.ref_bias_a = center
        self.ref_bias_phase = phi0
        self.trajectory = self.generate_ref_trajectory(self.ref_amplitude, self.ref_period, self.ref_bias_a, self.ref_bias_phase)

    def generate_random_start_target(self):
        x = np.random.uniform(low=self.pos_zone[0][0], high=self.pos_zone[0][1], size=2)
        y = np.random.uniform(low=self.pos_zone[1][0], high=self.pos_zone[1][1], size=2)
        z = np.random.uniform(low=self.pos_zone[2][0], high=self.pos_zone[2][1], size=2)
        psi = np.random.uniform(low=self.att_zone[2][0], high=self.att_zone[2][1], size=2)
        st = np.vstack((x, y, z, psi))
        start = st[:, 0]
        target = st[:, 1]
        return start, target

    def controller_reset(self):
        self.att_ctrl.fntsmc_att_reset()
        self.pos_ctrl.fntsmc_pos_reset()

    def controller_reset_with_new_param(self, new_att_param: fntsmc_param = None, new_pos_param: fntsmc_param = None):
        if new_att_param is not None:
            self.att_ctrl.fntsmc_att_reset_with_new_param(new_att_param)
        if new_pos_param is not None:
            self.pos_ctrl.fntsmc_pos_reset_with_new_param(new_pos_param)

    def collector_reset(self, N: int):
        self.collector.reset(N)

    @staticmethod
    def set_random_init_pos(pos0: np.ndarray, r: np.ndarray):
        """
        @brief:         为无人机设置随机的初始位置
        @param pos0:    参考轨迹第一个点
        @param r:       容许半径
        @return:        无人机初始点
        """
        return np.random.uniform(low=pos0 - np.fabs(r), high=pos0 + np.fabs(r), size=3)

    def generate_action_4_uav(self, att_limit: bool = True):
        ref, dot_ref, dot2_ref, _ = ref_uav(self.time, self.ref_amplitude, self.ref_period, self.ref_bias_a, self.ref_bias_phase)

        phi_d, theta_d, throttle = self.pos_control(ref[0:3], dot_ref[0:3], dot2_ref[0:3])
        dot_phi_d = (phi_d - self.att_ref[0]) / self.dt
        dot_theta_d = (theta_d - self.att_ref[1]) / self.dt

        rho_d = np.array([phi_d, theta_d, ref[3]])
        dot_rho_d = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])

        '''期望角速度限制'''
        if att_limit:
            dot_rho_d = np.clip(dot_rho_d, -self.dot_att_ref_limit, self.dot_att_ref_limit)
            rho_d += dot_rho_d * self.dt
        '''期望角速度限制'''

        torque = self.att_control(rho_d, dot_rho_d, np.zeros(3), att_only=False)
        action_4_uav = [throttle, torque[0], torque[1], torque[2]]

        return action_4_uav

    def reset_uav_pos_ctrl(self,
                           random_trajectory: bool = False,
                           random_pos0: bool = False,
                           yaw_fixed: bool = False,
                           new_att_ctrl_param: fntsmc_param = None,
                           new_pos_ctrl_parma: fntsmc_param = None,
                           outer_param: list = None):
        """
        @param outer_param:
        @param yaw_fixed:
        @param random_trajectory:
        @param random_pos0:
        @param new_att_ctrl_param:
        @param new_pos_ctrl_parma:
        @return:
        """
        '''1. generate random trajectory'''
        self.generate_random_trajectory(is_random=random_trajectory, yaw_fixed=yaw_fixed, outer_param=outer_param)

        '''2. reset uav randomly or not'''
        if random_pos0:
            _param = self.get_param_from_uav()
            _param.pos0 = self.set_random_init_pos(pos0=self.trajectory[0][0:3], r=0.3 * np.ones(3))
            self.reset_uav_with_param(_param)
        else:
            self.reset_uav()

        '''3. reset collector'''
        self.collector_reset(round(self.time_max / self.dt))

        '''4. reset controller'''
        if new_att_ctrl_param is not None:
            self.att_ctrl.fntsmc_att_reset_with_new_param(new_att_ctrl_param)
        else:
            self.att_ctrl.fntsmc_att_reset()

        if new_pos_ctrl_parma is not None:
            self.pos_ctrl.fntsmc_pos_reset_with_new_param(new_pos_ctrl_parma)
        else:
            self.pos_ctrl.fntsmc_pos_reset()

        '''5. reset image'''
        self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
        self.image_copy = self.image.copy()
        self.draw_3d_trajectory_projection(self.trajectory)
        self.draw_init_image()
