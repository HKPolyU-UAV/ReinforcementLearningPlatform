import numpy as np

from uav import UAV
from environment.envs.PIDControl.pid import PID
from common.common_func import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from uav_visualization import UAV_Visualization
import math

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as animation

'''位置画图相关变量'''
# fig, ax = plt.subplots(figsize=(30, 7), ncols=3, nrows=2)
# '''X'''
# plt_traj_ref_x, = ax[0][0].plot([], [], color='blue', linewidth=1)  # 参考轨迹 x
# plt_cur_pos_ref_x, = ax[0][0].plot([], [], marker='o', color='blue', markersize=5)  # 当前参考点 x
# plt_traj_uav_x, = ax[0][0].plot([], [], color='red', linewidth=1)  # 实际轨迹 x
# plt_cur_pos_uav_x, = ax[0][0].plot([], [], marker='o', color='red', markersize=5)  # 当前实际点 x
#
# '''Y'''
# plt_traj_ref_y, = ax[0][1].plot([], [], color='blue', linewidth=1)  # 实际轨迹 y
# plt_cur_pos_ref_y, = ax[0][1].plot([], [], marker='o', color='blue', markersize=5)  # 当前实际点 xy
# plt_traj_uav_y, = ax[0][1].plot([], [], color='red', linewidth=1)  # 实际轨迹 x
# plt_cur_pos_uav_y, = ax[0][1].plot([], [], marker='o', color='red', markersize=5)  # 当前实际点 x
#
# '''Z'''
# plt_traj_ref_z, = ax[0][2].plot([], [], color='blue', linewidth=1)  # 参考轨迹 z
# plt_cur_pos_ref_z, = ax[0][2].plot([], [], marker='o', color='blue', markersize=5)  # 当前参考点 z
# plt_traj_uav_z, = ax[0][2].plot([], [], color='red', linewidth=1)  # 实际轨迹 z
# plt_cur_pos_uav_z, = ax[0][2].plot([], [], marker='o', color='red', markersize=5)  # 当前实际点 z
#
# '''phi'''
# plt_phi_ref, = ax[1][0].plot([], [], color='blue', linewidth=1)  # phi 参考
# plt_cur_phi_ref, = ax[1][0].plot([], [], marker='o', color='blue', markersize=5)  # phi 当前参考
# plt_phi_uav, = ax[1][0].plot([], [], color='red', linewidth=1)
# plt_cur_phi_uav, = ax[1][0].plot([], [], marker='o', color='red', markersize=5)
#
# '''theta'''
# plt_theta_ref, = ax[1][1].plot([], [], color='blue', linewidth=1)
# plt_cur_theta_ref, = ax[1][1].plot([], [], marker='o', color='blue', markersize=5)
# plt_theta_uav, = ax[1][1].plot([], [], color='red', linewidth=1)
# plt_cur_theta_uav, = ax[1][1].plot([], [], marker='o', color='red', markersize=5)
#
# '''psi'''
# plt_psi_ref, = ax[1][2].plot([], [], color='blue', linewidth=1)
# plt_cur_psi_ref, = ax[1][2].plot([], [], marker='o', color='blue', markersize=5)
# plt_psi_uav, = ax[1][2].plot([], [], color='red', linewidth=1)
# plt_cur_psi_uav, = ax[1][2].plot([], [], marker='o', color='red', markersize=5)

'''姿态画图相关变量'''

# def plot_pre_config():
# 	fig.set_tight_layout(True)
# 	ax[0][0].set_ylim((quad.xmin, quad.xmax))
# 	ax[0][0].yaxis.set_major_locator(MultipleLocator(2))
# 	ax[0][0].set_title('x(t)')
#
# 	ax[0][1].set_ylim((quad.ymin, quad.ymax))
# 	ax[0][1].yaxis.set_major_locator(MultipleLocator(2))
# 	ax[0][1].set_title('y(t)')
#
# 	ax[0][2].set_ylim((quad.zmin, quad.zmax))
# 	ax[0][2].yaxis.set_major_locator(MultipleLocator(2))
# 	ax[0][2].set_title('z(t)')
#
# 	ax[1][0].set_ylim((rad2deg(quad.phimin), rad2deg(quad.phimax)))
# 	ax[1][0].yaxis.set_major_locator(MultipleLocator(20))
# 	ax[1][0].set_title('$\phi(t)$')
#
# 	ax[1][1].set_ylim((rad2deg(quad.thetamin), rad2deg(quad.thetamax)))
# 	ax[1][1].yaxis.set_major_locator(MultipleLocator(20))
# 	ax[1][1].set_title('$\Theta(t)$')
#
# 	ax[1][2].set_ylim((rad2deg(quad.psimin), rad2deg(quad.psimax)))
# 	ax[1][2].yaxis.set_major_locator(MultipleLocator(30))
# 	ax[1][2].set_title('$\psi(t)$')


if __name__ == '__main__':
    pos0 = np.array([0, 4, 0])  # 给的都是惯性系 东北天
    angle0 = np.array([deg2rad(0), deg2rad(0), deg2rad(0)])  # 给的都是惯性系 东北天

    quad = UAV(pos0=pos0, angle0=angle0)  # initialization of a quadrotor

    xbound = np.array([quad.pos_min[0], quad.pos_max[0]])
    ybound = np.array([quad.pos_min[1], quad.pos_max[1]])
    zbound = np.array([quad.pos_min[2], quad.pos_max[2]])
    origin = quad.pos

    quad_vis = UAV_Visualization(xbound=xbound, ybound=ybound, zbound=zbound, origin=origin, target=np.array([0, 0, 0]))  # 初始化显示界面
    quad_vis.arm_scale = 10

    '''visualization'''
    traj_ref = np.atleast_2d([[]])
    traj_uav = np.atleast_2d([[]])
    att_ref = np.atleast_2d([[]])
    att_uav = np.atleast_2d([[]])
    t = np.array([])
    count = 0
    # plot_pre_config()
    '''visualization'''

    inv_coe_m = np.linalg.inv(quad.power_allocation_mat)  # 动力分配矩阵的逆

    pid_x = PID(kp=0.5, ki=0., kd=120)  # controller of x
    pid_y = PID(kp=0.5, ki=0., kd=120)  # controller of y
    pid_z = PID(kp=0.5, ki=0., kd=160)  # controller z

    pid_phi = PID(kp=0.5, ki=0., kd=20)  # controller of roll along X in world
    pid_theta = PID(kp=0.5, ki=0., kd=20)  # controller of pitch along Y in world
    pid_psi = PID(kp=0.1, ki=0., kd=10)  # controller of yaw along Y in world

    plt.ion()
    while not quad.is_episode_Terminal():
        '''1. 生成参考轨迹'''
        phase = 2 * math.pi / 10 * quad.time
        '''八字'''
        # 初值 [0, 4, 0]
        x_ref = 6 * math.sin(phase) * math.cos(phase) / (1 + math.sin(phase) ** 2)
        y_ref = 4 * math.cos(phase) / (1 + math.sin(phase) ** 2)
        z_ref = 1 * math.sin(phase)
        psi_ref = deg2rad(0) * math.sin(math.pi * quad.time)

        '''圆'''
        # # 初值 [0, 5, 0]
        # x_ref = 5 * math.sin(phase)
        # y_ref = 5 * math.cos(phase)
        # z_ref = 3 * math.sin(phase)
        # psi_ref = deg2rad(45) * math.sin(phase)
        # x_ref = 0
        # y_ref = 0
        # z_ref = 0

        t_ref = quad.time
        quad_vis.target = np.array([x_ref, y_ref, z_ref])
        quad_vis.has_target = True
        quad_vis.has_ref = False

        '''2. 计算位置 PID 控制的输出，同时得到期望姿态角'''
        ex, ey, ez = x_ref - quad.pos[0], y_ref - quad.pos[1], z_ref - quad.pos[2]
        pid_x.set_e(ex)
        pid_y.set_e(ey)
        pid_z.set_e(ez)
        ux, uy, uz = pid_x.out(), pid_y.out(), pid_z.out()
        U1 = quad.m * math.sqrt(ux ** 2 + uy ** 2 + (uz + quad.g) ** 2)

        psi_ref = deg2rad(0)*quad.time
        if psi_ref > 2*np.pi:
            psi_ref -= (int)(psi_ref / 2 / np.pi) * 2 * np.pi
        if psi_ref < -2*np.pi:
            psi_ref += (int)(psi_ref / 2 / np.pi) * 2 * np.pi
        # phi_ref = math.asin((ux * math.sin(psi_ref) - uy * math.cos(psi_ref)) * quad.m / U1)
        # theta_ref = math.asin((ux * quad.m - U1 * math.sin(psi_ref) * math.sin(phi_ref)) / (U1 * math.cos(psi_ref) * math.cos(phi_ref)))
        phi_ref = np.arcsin(quad.m * (ux * np.sin(psi_ref) - uy * np.cos(psi_ref)) / U1)
        theta_ref = np.arcsin(quad.m * (ux * np.cos(psi_ref) + uy * np.sin(psi_ref)) / (U1 * np.cos(phi_ref)))

        # phi_ref = deg2rad(30) * math.sin(2 * math.pi / 1 * quad.time)
        # theta_ref = deg2rad(30) * math.sin(2 * math.pi / 1 * quad.time)

        # print('期望姿态角：', phi_ref, theta_ref, psi_ref)

        '''3. 绘制图形 (图形必须要先画，不然的话无人机状态和参考轨迹的状态差一拍)'''
        if count < 6000:
            if count == 0:
                traj_ref = np.atleast_2d(np.array([x_ref, y_ref, z_ref]))
                traj_uav = np.atleast_2d([quad.pos])
                att_ref = np.atleast_2d(np.array([phi_ref, theta_ref, psi_ref]))
                att_uav = np.atleast_2d([quad.angle])
                t = np.array([quad.time])
            else:
                traj_ref = np.vstack((traj_ref, np.array([x_ref, y_ref, z_ref])))
                traj_uav = np.vstack((traj_uav, quad.pos))
                att_ref = np.vstack((att_ref, np.array([phi_ref, theta_ref, psi_ref])))
                att_uav = np.vstack((att_uav, quad.angle))
                t = np.vstack((t, quad.time))
        else:
            traj_ref = np.delete(traj_ref, [-1], axis=0)
            traj_uav = np.delete(traj_ref, [-1], axis=0)
            att_ref = np.delete(traj_ref, [-1], axis=0)
            att_uav = np.delete(traj_ref, [-1], axis=0)
            t = np.delete(t, [-1])

            traj_ref = np.vstack((traj_ref, np.array([x_ref, y_ref, z_ref])))
            traj_uav = np.vstack((traj_uav, quad.pos))
            att_ref = np.vstack((att_ref, np.array([phi_ref, theta_ref, psi_ref])))
            att_uav = np.vstack((att_uav, quad.angle))
            t = np.vstack((t, quad.time))
        count += 1

        plot_x = False
        plot_y = False
        plot_z = False
        plot_phi = False
        plot_theta = False
        plot_psi = False
        plot_3D = True

        # if count % 10 == 0:
        # 	if plot_x:
        # 		ax[0][0].set_xlim((t[0], t_ref)) if t_ref > t[0] else ax[0].set_xlim((t[0], t[0] + quad.dt))
        # 		plt_traj_ref_x.set_data(t, traj_ref[0])
        # 		plt_cur_pos_ref_x.set_data([t_ref], [x_ref])
        # 		plt_traj_uav_x.set_data(t, traj_uav[0])
        # 		plt_cur_pos_uav_x.set_data([t_ref], [quad.x])
        #
        # 	if plot_y:
        # 		ax[0][1].set_xlim((t[0], t_ref)) if t_ref > t[0] else ax[1].set_xlim((t[0], t[0] + quad.dt))
        # 		plt_traj_ref_y.set_data(t, traj_ref[1])
        # 		plt_cur_pos_ref_y.set_data([t_ref], [y_ref])
        # 		plt_traj_uav_y.set_data(t, traj_uav[1])
        # 		plt_cur_pos_uav_y.set_data([t_ref], [quad.y])
        #
        # 	if plot_z:
        # 		ax[0][2].set_xlim((t[0], t_ref)) if t_ref > t[0] else ax[2].set_xlim((t[0], t[0] + quad.dt))
        # 		plt_traj_ref_z.set_data(t, traj_ref[2])
        # 		plt_cur_pos_ref_z.set_data([t_ref], [z_ref])
        # 		plt_traj_uav_z.set_data(t, traj_uav[2])
        # 		plt_cur_pos_uav_z.set_data([t_ref], [quad.z])
        #
        # 	if plot_phi:
        # 		ax[1][0].set_xlim((t[0], t_ref))if t_ref > t[0] else ax[3].set_xlim((t[0], t[0] + quad.dt))
        # 		plt_phi_ref.set_data(t, att_ref[0])
        # 		plt_cur_phi_ref.set_data([t_ref], [phi_ref])
        # 		plt_phi_uav.set_data(t, att_uav[0])
        # 		plt_cur_phi_uav.set_data([t_ref], [quad.phi])
        #
        # 	if plot_theta:
        # 		ax[1][1].set_xlim((t[0], t_ref)) if t_ref > t[0] else ax[4].set_xlim((t[0], t[0] + quad.dt))
        # 		plt_theta_ref.set_data(t, att_ref[1])
        # 		plt_cur_theta_ref.set_data([t_ref], [theta_ref])
        # 		plt_theta_uav.set_data(t, att_uav[1])
        # 		plt_cur_theta_uav.set_data([t_ref], [quad.theta])
        #
        # 	if plot_psi:
        # 		ax[1][2].set_xlim((t[0], t_ref)) if t_ref > t[0] else ax[5].set_xlim((t[0], t[0] + quad.dt))
        # 		plt_psi_ref.set_data(t, att_ref[2])
        # 		plt_cur_psi_ref.set_data([t_ref], [psi_ref])
        # 		plt_psi_uav.set_data(t, att_uav[2])
        # 		plt_cur_psi_uav.set_data([t_ref], [quad.psi])

        if plot_3D:
            quad_vis.render(p=quad.pos,
                            ref_p=np.array([x_ref, y_ref, z_ref]),
                            v=quad.vel,
                            a=quad.angle,
                            ra=quad.omega_inertial,
                            f=quad.force, d=quad.d, win=10)

        plt.show()
        # canvas = FigureCanvasAgg(plt.gcf())

        plt.pause(0.0000000001)

        '''4. 计算姿态 PID 的控制输出'''
        e_phi, e_theta, e_psi = phi_ref - quad.angle[0], theta_ref - quad.angle[1], psi_ref - quad.angle[2]
        e_psi1 = psi_ref - 2 * np.pi - quad.angle[2]    # 一定为负
        e_psi2 = psi_ref + 2 * np.pi - quad.angle[2]    # 一定为正
        v = np.array([e_psi, e_psi1, e_psi2])
        s = np.sign(v)  # 三个变量的符号
        index = np.argmin(np.fabs(v))
        e_psi = v[index]

        pid_phi.set_e(e_phi)
        pid_theta.set_e(e_theta)
        pid_psi.set_e(e_psi)
        U2, U3, U4 = pid_phi.out(), pid_theta.out(), pid_psi.out()

        '''5. 动力分配'''
        square_omega = np.maximum(np.dot(inv_coe_m, [U1, U2, U3, U4]), 0)
        f = quad.CT * square_omega
        f = quad.fmax * np.tanh(f / quad.fmax)
        # print('PID out: U1: %.3f, U2: %.3f,U3: %.3f, U4: %.3f' % (U1, U2, U3, U4))
        # for i in range(4):
        #     f[i] = max(min(quad.fmax, f[i]), quad.fmin)
        # print('Force UAV: ', f)
        # print('time', quad.time)
        quad.rk44(action=f)

    plt.ioff()
