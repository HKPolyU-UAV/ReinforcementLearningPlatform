import numpy as np

from uav import UAV
from uav_visualization import UAV_Visualization
from common.common_func import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
	pos0 = np.array([0, 0, 0])	# 给的都是惯性系 东北天
	angle0 = np.array([deg2rad(0), deg2rad(0), deg2rad(0)])  # 给的都是惯性系 东北天

	quad = UAV(pos0=pos0, angle0=angle0)
	# quad.set_position_limitation2inf()

	xbound = np.array([quad.pos_min[0], quad.pos_max[0]])
	ybound = np.array([quad.pos_min[1], quad.pos_max[1]])
	zbound = np.array([quad.pos_min[2], quad.pos_max[2]])
	origin = quad.pos

	quad_vis = UAV_Visualization(xbound=xbound, ybound=ybound, zbound=zbound, origin=origin, target=np.array([5, 5, 5]))  # 初始化显示界面
	index = 0
	plt.ion()
	while not quad.is_episode_Terminal():
		# position = quad.pos
		# attitude = quad.angle
		d = 4 * quad.d

		'''一些常规力选择'''
		f0 = 9.8 / 5
		eq_f = quad.m * quad.g / math.cos(math.fabs(quad.angle[0])) / math.cos(math.fabs(quad.angle[1])) / 4
		bias = 1 * math.sin(math.pi * quad.time + math.pi / 2)

		f_roll = [f0 - 0.02, f0, f0, f0 - 0.02]  # 滚转X
		f_pitch = [f0 - 0.02, f0 - 0.02, f0, f0]  # 俯仰Y
		f_yaw = [f0 + 0.02, f0 - 0.02, f0 + 0.02, f0 - 0.02]  # 偏航Z
		f_balance = [f0, f0, f0, f0]  # 平衡
		f_free_fall = [0, 0, 0, 0]  # 自由落体
		f_down = [f0 - 0.1, f0 - 0.1, f0 - 0.1, f0 - 0.1]  # 下降
		f_up = [f0 + 0.1, f0 + 0.1, f0 + 0.1, f0 + 0.1]  # 上升
		f_keep_att = [eq_f for _ in range(4)]  # 保持姿态不变
		f_sin = [f0 + bias, f0 + bias, f0 + bias, f0 + bias]  # 正弦平动
		f = f_down
		'''一些常规力选择'''

		'''visualization'''
		quad_vis.render(p=quad.pos, ref_p=np.array([0, 0, 10]), v=quad.vel, a=quad.angle, ra=quad.omega_inertial, d=d, f=np.array(f), win=10)
		if index % 50 == 0:
			quad.show_uav_linear_state(with_time=True)
			# quad.show_uav_angular_state(with_time=True)
		plt.show()
		plt.pause(0.0000000001)
		'''visualization'''

		action = f  # 给无人机四个螺旋桨的力
		quad.rk44(action=np.array(action))
		index += 1
	plt.ioff()
	plt.show()
