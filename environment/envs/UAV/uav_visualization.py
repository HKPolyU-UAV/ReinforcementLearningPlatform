import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.pyplot import MultipleLocator
# from matplotlib.axes import Axes
import numpy as np
import math


class UAV_Visualization:
    def __init__(self, xbound: np.ndarray, ybound: np.ndarray, zbound: np.ndarray, origin: np.ndarray, target: np.ndarray, arm_scale: float = 5):
        """
        :param xbound:      观察系的
        :param ybound:      观察系的
        :param zbound:      观察系的
        :param origin:      观察系的
        """
        self.fig = plt.figure(figsize=(9, 9))
        # self.ims = []
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.o = origin
        self.target = target
        self.arm_scale = arm_scale
        self.ax = axes3d.Axes3D(self.fig)
        self.ax.set_aspect('auto')      # 只能auto
        self.is_dynamic_axis = max(np.max(np.fabs(xbound)), np.max(np.fabs(ybound)), np.max(np.fabs(zbound))) >= math.inf
        if not self.is_dynamic_axis:        # 动态坐标轴
            self.ax.set_xlim3d(self.xbound)
            self.ax.set_ylim3d(self.ybound)
            self.ax.set_zlim3d(self.zbound)
        else:
            self.ax.set_xlim3d([self.o[0] - xbound[0] / 2, self.o[0] + xbound[1] / 2])
            self.ax.set_ylim3d([self.o[1] - ybound[0] / 2, self.o[1] + ybound[1] / 2])
            self.ax.set_zlim3d([self.o[2] - zbound[0] / 2, self.o[2] + zbound[1] / 2])
        self.ax.xaxis.set_major_locator(MultipleLocator(2))
        self.ax.yaxis.set_major_locator(MultipleLocator(2))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('QuadrotorFly Simulation', fontsize='13')

        self.traj_ref_data = np.atleast_2d([])
        self.traj_data = np.atleast_2d(self.o)
        self.traj_count = 1000
        self.sim_index = 0
        self.length_per_n = 0.6
        self.color = ['red', 'orange', 'blue', 'black']

        self.has_ref = False        # 默认是没有轨迹跟踪的
        self.has_target = True      # 默认有一个目标点

        '''UAV相关部件'''
        self.origin_point = self.ax.plot([], [], [], marker='o', color='black', markersize=6, antialiased=False)[0]   # 画原点
        self.target_point = self.ax.plot([], [], [], marker='o', color='red', markersize=10, antialiased=False)[0]    # 画终点
        self.center = self.ax.plot([], [], [], marker='o', color='blue', markersize=6, antialiased=False)[0]          # 中心
        self.bar = [self.ax.plot([], [], [], color=self.color[i], linewidth=2.5, antialiased=False)[0] for i in range(4)]
        self.bar_ball = [self.ax.plot([], [], [], marker='o', color=self.color[i], markersize=6, antialiased=False)[0] for i in range(4)]
        self.head = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)[0]           # 机头点
        self.head_bar = self.ax.plot([], [], [], color='green', markersize=6, antialiased=False)[0]                   # 飞机正方向
        self.traj = self.ax.plot([], [], [], color='red', linewidth=1.5)[0]
        self.traj_ref = self.ax.plot([], [], [], color='blue', linewidth=1.5)[0]
        self.X_proj = self.ax.plot([], [], [], marker='o', color='red', markersize=6, antialiased=False)[0]
        self.Y_proj = self.ax.plot([], [], [], marker='o', color='red', markersize=6, antialiased=False)[0]
        self.Z_proj = self.ax.plot([], [], [], marker='o', color='red', markersize=6, antialiased=False)[0]
        self.text_e = self.ax.text2D(0.02, 0.92 - 0.00, 'Pos_e: {} m'.format([0, 0, 0]), transform=self.ax.transAxes, fontsize='11')
        self.text_pos = self.ax.text2D(0.02, 0.92 - 0.03, 'Pos: {} m'.format([0, 0, 0]), transform=self.ax.transAxes, fontsize='11')
        self.text_vel = self.ax.text2D(0.02, 0.92 - 0.06, 'Pos: {} m/s'.format([0, 0, 0]), transform=self.ax.transAxes, fontsize='11')
        self.text_att = self.ax.text2D(0.02, 0.92 - 0.09, 'Pos: {} deg'.format([0, 0, 0]), transform=self.ax.transAxes, fontsize='11')
        self.text_att_rate = self.ax.text2D(0.02, 0.92 - 0.12, 'Pos: {} deg/s'.format([0, 0, 0]), transform=self.ax.transAxes, fontsize='11')

        self.label = [self.ax.text(self.o[0], self.o[1], self.o[2], str(i + 1), fontsize='11') for i in range(4)]
        self.grav = self.ax.quiver(self.o[0], self.o[1], self.o[2], 0, 0, -1, length=0.8*9.8*self.length_per_n, color='red')

        self.rotor_force = [self.ax.quiver(self.o[0], self.o[1], self.o[2], 0, 0, 0, length=1, color=self.color[i]) for i in range(4)]

        self.quadGui = {
            'origin_point': self.origin_point,
            'target_point': self.target_point,
            'center': self.center,
            'bar': self.bar,
            'head': self.head,
            'head_bar': self.head_bar,
            'traj': self.traj,
            'traj_ref': self.traj_ref,
            'bar_ball': self.bar_ball,
            'X_proj': self.X_proj,
            'Y_proj': self.Y_proj,
            'Z_proj': self.Z_proj,
            'text_e': self.text_e,
            'text_pos': self.text_pos,
            'text_vel': self.text_vel,
            'text_att': self.text_att,
            'text_att_rate': self.text_att_rate
        }
        '''UAV相关部件'''

        self.figure = self.ax.plot([], [])
        self.draw_bound()

    def draw_bound(self):
        """
        :return:        绘制外框
        """
        for _z in self.zbound:
            for ik in [0, 1]:
                posx = self.xbound
                posy = [self.ybound[ik], self.ybound[ik]]
                posz = [_z, _z]
                self.figure = self.ax.plot(posx, posy, posz, 'y')

                posx = [self.xbound[ik], self.xbound[ik]]
                posy = self.ybound
                posz = [_z, _z]
                self.figure = self.ax.plot(posx, posy, posz, 'y')
        for _x in self.xbound:
            for _y in self.ybound:
                self.figure = self.ax.plot([_x, _x], [_y, _y], self.zbound, 'y')

    @staticmethod
    def rotate_matrix(attitude: np.ndarray):
        [phi, theta, psi] = attitude
        _R_i_b1 = np.array([[math.cos(psi), math.sin(psi), 0],
                            [-math.sin(psi), math.cos(psi), 0],
                            [0, 0, 1]])  # 从惯性系到b1系，旋转偏航角psi
        _R_b1_b2 = np.array([[math.cos(theta), 0, -math.sin(theta)],
                             [0, 1, 0],
                             [math.sin(theta), 0, math.cos(theta)]])  # 从b1系到b2系，旋转俯仰角theta
        _R_b2_b = np.array([[1, 0, 0],
                            [0, math.cos(phi), math.sin(phi)],
                            [0, -math.sin(phi), math.cos(phi)]])  # 从b2系到b系，旋转滚转角phi
        _R_i_b = np.matmul(_R_b2_b, np.matmul(_R_b1_b2, _R_i_b1))  # 从惯性系到机体系的转换矩阵
        _R_b_i = _R_i_b.T  # 从机体系到惯性系的转换矩阵
        return _R_b_i

    def render(self,
               p: np.ndarray,
               ref_p: np.ndarray,
               v: np.ndarray,
               a: np.ndarray,
               ra: np.ndarray,
               f: np.ndarray,
               d: float,
               win: int):
        """
        @note:          show
        @param p:       无人机在 观察系 下的位置
        @param ref_p:   reference position (target)
        @param v:       无人机速度
        @param a:       无人机在 观察系 下的姿态
        @param ra:      无人机姿态角速度
        @param f:       无人机螺旋桨的升力
        @param d:       无人机机臂长度
        @param win:     时间窗口长度
        @return:        NOne
        """
        '''轨迹数据存储'''
        if self.sim_index < self.traj_count:
            self.traj_data = np.vstack((self.traj_data, p))
            if self.has_ref:
                if self.sim_index == 0:
                    self.traj_ref_data = np.atleast_2d(ref_p)
                else:
                    self.traj_ref_data = np.vstack((self.traj_ref_data, ref_p))
        else:
            self.traj_data = np.delete(self.traj_data, [0], axis=0)
            if self.has_ref:
                self.traj_ref_data = np.delete(self.traj_ref_data, [0], axis=0)

            self.traj_data = np.vstack((self.traj_data, p))
            if self.has_ref:
                self.traj_ref_data = np.vstack((self.traj_ref_data, ref_p))
        '''轨迹数据存储'''

        if self.sim_index % win == 0:
            R_b_i = self.rotate_matrix(a)
            d0 = self.arm_scale * d / math.sqrt(2)
            bar = np.dot(R_b_i, np.atleast_2d([[d0, d0, 0], [d0, -d0, 0], [-d0, -d0, 0], [-d0, d0, 0]]).T) + np.vstack((p, p, p, p)).T
            label = np.dot(R_b_i, np.atleast_2d([[d0, d0, 0.3], [d0, -d0, 0.3], [-d0, -d0, 0.3], [-d0, d0, 0.3]]).T) + np.vstack((p, p, p, p)).T
            head = np.dot(R_b_i, [2 * d0, 0, 0]) + p
            dir_f = np.dot(R_b_i, [0, 0, 1])

            if self.is_dynamic_axis:
                self.ax.set_xlim3d([p[0] - 10, p[0] + 10])
                self.ax.set_ylim3d([p[1] - 10, p[1] + 10])
                self.ax.set_zlim3d([p[2] - 10, p[2] + 10])

            '''初始位置'''
            self.quadGui['origin_point'].set_data([self.o[0], self.o[0]], [self.o[1], self.o[1]])
            self.quadGui['origin_point'].set_3d_properties([self.o[2], self.o[2]])
            '''初始位置'''

            '''文字'''
            np.set_printoptions(precision=3)
            self.quadGui['text_e'].set_text('Pos_e: {} m'.format(self.target - p))
            self.quadGui['text_pos'].set_text('Pos: {} m'.format(p))
            self.quadGui['text_vel'].set_text('Vel: {} m/s'.format(v))
            self.quadGui['text_att'].set_text('Att: {} deg'.format(a * 180 / np.pi))
            self.quadGui['text_att_rate'].set_text('ARate: {} deg/s'.format(ra * 180 / np.pi))
            '''文字'''

            '''目标位置 (如果有)'''
            if self.has_target:
                self.quadGui['target_point'].set_data([self.target[0], self.target[0]], [self.target[1], self.target[1]])
                self.quadGui['target_point'].set_3d_properties([self.target[2], self.target[2]])
            '''目标位置 (如果有)'''

            '''无人机中心位置'''
            self.quadGui['center'].set_data([p[0], p[0]], [p[1], p[1]])
            self.quadGui['center'].set_3d_properties([p[2], p[2]])
            '''无人机中心位置'''

            '''无人机位置在三轴投影'''
            print('x', self.xbound)
            print('y', self.ybound)
            print('z', self.zbound)
            self.quadGui['X_proj'].set_data([p[0], p[0]], [self.ybound[0], self.ybound[0]])
            self.quadGui['X_proj'].set_3d_properties([self.zbound[0], self.zbound[0]])
            self.quadGui['Y_proj'].set_data([self.xbound[1], self.xbound[1]], [p[1], p[1]])
            self.quadGui['Y_proj'].set_3d_properties([self.zbound[0], self.zbound[0]])
            self.quadGui['Z_proj'].set_data([self.xbound[1], self.xbound[1]], [self.ybound[1], self.ybound[1]])
            self.quadGui['Z_proj'].set_3d_properties([p[2], p[2]])
            '''无人机位置在三轴投影'''

            '''画标签'''
            for i in range(4):
                self.label[i].remove()
                self.label[i] = self.ax.text(label[:, i][0], label[:, i][1], label[:, i][2], str(i + 1), fontsize='11')
            '''画标签'''

            '''四个机臂位置'''
            for i in range(4):
                self.quadGui['bar'][i].set_data([bar[:, i][0], p[0]], [bar[:, i][1], p[1]])
                self.quadGui['bar'][i].set_3d_properties([bar[:, i][2], p[2]], [bar[:, i][2], p[2]])
                self.quadGui['bar_ball'][i].set_data([bar[:, i][0], bar[:, i][0]], [bar[:, i][1], bar[:, i][1]])
                self.quadGui['bar_ball'][i].set_3d_properties([bar[:, i][2], bar[:, i][2]], [bar[:, i][2], bar[:, i][2]])
            '''四个机臂位置'''

            '''飞机正方向'''
            self.quadGui['head_bar'].set_data([head[0], p[0]], [head[1], p[1]])
            self.quadGui['head_bar'].set_3d_properties([head[2], p[2]])
            self.quadGui['head'].set_data([head[0], head[0]], [head[1], head[1]])
            self.quadGui['head'].set_3d_properties([head[2], head[2]])
            '''飞机正方向'''

            '''轨迹'''
            self.quadGui['traj'].set_data(self.traj_data[:, 0], self.traj_data[:, 1])
            self.quadGui['traj'].set_3d_properties(self.traj_data[:, 2])
            '''轨迹'''

            '''参考轨迹 (如果有)'''
            if self.has_ref:
                self.quadGui['traj_ref'].set_data(self.traj_ref_data[:, 0], self.traj_ref_data[:, 1])
                self.quadGui['traj_ref'].set_3d_properties(self.traj_ref_data[:, 2])
            '''参考轨迹 (如果有)'''

            '''重力加速度'''
            self.grav.remove()
            self.grav = self.ax.quiver(p[0], p[1], p[2], 0, 0, -1, length=0.8*9.8*self.length_per_n, color='black')
            '''重力加速度'''

            '''升力'''
            for i in range(4):
                self.rotor_force[i].remove()
                self.rotor_force[i] = self.ax.quiver(bar[:, i][0], bar[:, i][1], bar[:, i][2],
                                                     dir_f[0], dir_f[1], dir_f[2],
                                                     length=f[i]*self.length_per_n, color=self.color[i])
            '''升力'''
        self.sim_index += 1

    def reset(self, origin: np.ndarray, target: np.ndarray):
        """
        @note:
        @param origin:
        @param target:
        @return:
        """
        self.o = origin
        self.target = target
        self.traj_ref_data = np.atleast_2d([])
        self.traj_data = np.atleast_2d(self.o)
        self.sim_index = 0
        self.has_ref = False
        self.has_target = True
        for i in range(4):
            self.label[i].remove()
            self.label[i] = self.ax.text(self.o[0], self.o[1], self.o[2], str(i), fontsize='11')
