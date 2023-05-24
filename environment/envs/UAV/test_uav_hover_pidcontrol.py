import math

from common.common_func import *
from uav_hover import UAV_Hover
import matplotlib.pyplot as plt
from environment.envs.PIDControl.pid import PID


if __name__ == '__main__':
    env = UAV_Hover(target_pos=[5, 5, 0], save_cfg=False)
    env.uav_vis.arm_scale = 10      # 显示放大的尺度，自己设置即可

    pid_x = PID(kp=0.5, ki=0., kd=120)  # controller of x
    pid_y = PID(kp=0.5, ki=0., kd=120)  # controller of y
    pid_z = PID(kp=0.7, ki=0., kd=200)  # controller z

    pid_phi = PID(kp=0.5, ki=0., kd=20)  # controller of roll along X in world
    pid_theta = PID(kp=0.5, ki=0., kd=20)  # controller of pitch along Y in world
    pid_psi = PID(kp=0.1, ki=0., kd=10)  # controller of yaw along Y in world

    num = 0

    inv_coe_m = np.linalg.inv(env.power_allocation_mat)  # 动力分配矩阵的逆

    plt.ion()
    while num < 10:
        env.reset_random()
        # env.reset()
        # env.reset_target_random()
        while not env.is_terminal:
            env.show_dynamic_image(per_show=10)
            plt.pause(0.00000001)
            '''control'''
            '''位置PID输出'''
            pid_x.set_e(env.error_pos[0])
            pid_y.set_e(env.error_pos[1])
            pid_z.set_e(env.error_pos[2])
            ux, uy, uz = pid_x.out(), pid_y.out(), pid_z.out()
            '''位置PID输出'''

            '''计算期望姿态'''
            U1 = env.m * math.sqrt(ux ** 2 + uy ** 2 + (uz + env.g) ** 2)
            psi_ref = deg2rad(1)
            phi_ref = np.arcsin(env.m * (ux * np.sin(psi_ref) - uy * np.cos(psi_ref)) / U1)
            theta_ref = np.arcsin(env.m * (ux * np.cos(psi_ref) + uy * np.sin(psi_ref)) / (U1 * np.cos(phi_ref)))
            '''计算期望姿态'''

            '''姿态PID输出'''
            e_phi, e_theta, e_psi = phi_ref - env.angle[0], theta_ref - env.angle[1], psi_ref - env.angle[2]
            pid_phi.set_e(e_phi)
            pid_theta.set_e(e_theta)
            pid_psi.set_e(e_psi)
            U2, U3, U4 = pid_phi.out(), pid_theta.out(), pid_psi.out()
            '''姿态PID输出'''

            '''动力分配'''
            square_omega = np.maximum(np.dot(inv_coe_m, [U1, U2, U3, U4]), 0)
            f = env.CT * square_omega
            f = env.fmax * np.tanh(f / env.fmax)
            '''动力分配'''
            '''control'''
            # f = [random.uniform(env.fmin, env.fmax) for _ in range(4)]
            env.step_update(action=f)
            print('Pos_e: {}'.format(env.error_pos))
        num += 1
    plt.ioff()
