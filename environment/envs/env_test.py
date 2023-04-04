import math
import os
import random
import sys
import cv2 as cv
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./")
from common.common_func import *


# Flight Attitude Simulator Test
def test_flight_attitude_simulator():
	from FlightAttitudeSimulator.flight_attitude_simulator import Flight_Attitude_Simulator
	env = Flight_Attitude_Simulator(initTheta=-60.0, setTheta=0., save_cfg=True)
	test_num = 1
	for _ in range(test_num):
		env.reset_random()
		while not env.is_terminal:
			env.show_dynamic_image(isWait=False)
			# action = random.choice(env.action_space)
			action = [0.0]
			env.step_update(action=action)
	# env.saveData(is2file=True, filepath='../../datasave/')


# Flight Attitude Simulator Continuous Test
def test_flight_attitude_simulator_continuous():
	from FlightAttitudeSimulator.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous
	env = Flight_Attitude_Simulator_Continuous(initTheta=-60.0, setTheta=0., save_cfg=True)
	test_num = 1
	for _ in range(test_num):
		# env.reset_random()
		env.reset()
		# env.theta=0
		while not env.is_terminal:
			env.show_dynamic_image(isWait=False)
			# action = random.choice(env.action_space)
			action = np.array([3])
			env.step_update(action=action)
			print(env.theta, env.dTheta)


def test_flight_attitude_simulator_2state_continuous():
	from FlightAttitudeSimulator.flight_attitude_simulator_2state_continuous import Flight_Attitude_Simulator_2State_Continuous as FAS_2S_Con
	env = FAS_2S_Con(initTheta=deg2rad(0.0), save_cfg=True)
	test_num = 1
	for _ in range(test_num):
		# env.reset_random()
		env.reset()
		# env.theta=0
		while not env.is_terminal:
			env.show_dynamic_image(isWait=True)
			# action = random.choice(env.action_space)
			action = np.array([3])
			env.step_update(action=action)
			# print(env.theta, env.dTheta)
		print(env.time)


# UGV Bidirectional Continuous Test
def test_ugv_bidirectional_continuous():
	from UGV.ugv_bidirectional_continuous import UGV_Bidirectional_Continuous
	env = UGV_Bidirectional_Continuous(initPhi=deg2rad(-135),
									   save_cfg=True,
									   x_size=10.0,
									   y_size=10.0,
									   start=[2.0, 2.0],
									   terminal=[4.0, 4.0])

	for _ in range(5):
		env.reset_random()
		while not env.is_terminal:
			# print(env.time)
			env.show_dynamic_image(isWait=False)
			action = [-3, 3]
			env.step_update(action=action)
			# print(env.reward)
			# print(env.current_state)


def test_UGVBidirectional2():
	from UGV2.UGVBidirectional import UGV_Bidirectional as env
	env = env(pos0=np.array([1.0, 1.0]),
			  phi0=deg2rad(0),
			  target=np.array([1.0, 4.0]),
			  map_size=np.array([5.0, 5.0]))  # 其余参数默认

	for _ in range(5):
		env.reset()
		# env.phi = -np.pi/2
		while not env.is_terminal:
			# print(env.time)
			env.show_dynamic_image(isWait=True)
			action = np.array([10, -10])
			env.step_update(action=action)
			print(rad2deg(env.sum_d_theta))
			# print(env.reward)
			# print(env.current_state)


def test_SecondOrderIntegration():
	from SecondOrderIntegration.SecondOrderIntegration import SecondOrderIntegration as env
	env = env(pos0=np.array([2.0, 2.0]), map_size=np.array([10.0, 10.0]))  # 其余参数默认

	for _ in range(5):
		env.reset()
		# env.phi = -np.pi/2
		while not env.is_terminal:
			# print(env.time)
			env.show_dynamic_image(isWait=True)
			action = np.array([-3, 2])
			env.step_update(action=action)
			print(np.hstack((env.error, env.pos, env.vel)))
			print(env.current_state)
			# print(env.vel)
			# print(env.reward)
			# print(env.current_state)


# UGV Forward Continuous Test
def test_ugv_forward_continuous():
	from UGV.ugv_forward_continuous import UGV_Forward_Continuous
	env = UGV_Forward_Continuous(initPhi=deg2rad(60),
								 save_cfg=True,
								 x_size=5.0,
								 y_size=5.0,
								 start=[2.5, 2.5],
								 terminal=[4.8, 4.8])
	while True:
		env.reset_random()
		# env.reset()
		env.show_dynamic_image(isWait=True)
		while not env.is_terminal:
			# print(env.time)
			if cv.waitKey(1) == 27:
				return
			env.show_dynamic_image(isWait=False)
			# action = [9, 6]
			action = env.towards_target_PID(threshold=10, kp=10, ki=0, kd=0)
			env.step_update(action=action)
			# print(env.reward)
			# print(env.current_state)


def test_UGVForward():
	from UGV2.UGVForward import UGV_Forward as env
	env = env(pos0=np.array([1.0, 1.0]),
			  vel0=np.array([0.0, 0.0]),
			  phi0=deg2rad(90),
			  omega0=0.,
			  map_size=np.array([5.0, 5.0]),
			  target=np.array([4.0, 4.0]))
	test_num = 5
	for _ in range(test_num):
		env.reset()
		sumr = 0
		while not env.is_terminal:
			# print(env.time)
			if cv.waitKey(1) == 27:
				return
			env.show_dynamic_image(isWait=True)
			action = np.array([-1., -2.])
			env.step_update(action=action)
			sumr += env.reward
		print(sumr)


# UGV Forward Obstacles Continuous Test
def test_ugv_forward_obstacles_continuous():
	from UGV.ugv_forward_obstacle_continuous import UGV_Forward_Obstacle_Continuous
	env = UGV_Forward_Obstacle_Continuous(initPhi=deg2rad(0), save_cfg=True, x_size=11, y_size=11, start=[2.5, 2.5], terminal=[4.0, 2.5],
										  dataBasePath='./pathplanning/11X11-AllCircle1/')
	num = 0
	while num < 1:
		# env.reset_random_with_database()
		env.reset_random()
		env.start = [1, 1]
		env.terminal = [9, 9]
		# env.show_dynamic_imagewithobs(isWait=False)
		# cv.imwrite(str(num) + '.png', env.image)

		while not env.is_terminal:
			# print(env.time)
			if cv.waitKey(1) == 27:
				return
			env.show_dynamic_imagewithobs(isWait=True)
			action = env.towards_target_PID(threshold=np.inf, kp=10, kd=0, ki=0)
			# cap.write(env.save)
			# action = [0, 1 * math.pi]
			env.step_update(action=action)
			# print(env.current_state[0 : 4])
			# if env.terminal_flag == 4:
			#     print(env.reward)
			# print(env.current _state)
		num += 1


# UGV Forward Discrete Test
def test_ugv_forward_discrete():
	from UGV.ugv_forward_discrete import UGV_Forward_Discrete
	env = UGV_Forward_Discrete(initPhi=deg2rad(60), save_cfg=True, x_size=5.0, y_size=5.0, start=[2.5, 2.5], terminal=[4.8, 4.8])
	while True:
		env.reset_random()
		env.show_dynamic_image(isWait=True)
		while not env.is_terminal:
			if cv.waitKey(1) == 27:
				return
			env.show_dynamic_image(isWait=False)
			action = env.towards_target_PID(threshold=10, kp=10, ki=0, kd=0)
			wLeft = random.choice(env.action_space[0])
			wRight = random.choice(env.action_space[1])
			action = [wLeft, wRight]
			env.step_update(action=action)


# UGV Forward OBstacle Discrete Test
def test_ugv_forward_obstacles_discrete():
	from UGV.ugv_forward_obstacle_discrete import UGV_Forward_Obstacle_Discrete
	env = UGV_Forward_Obstacle_Discrete(initPhi=deg2rad(0), save_cfg=True, x_size=11, y_size=11, start=[2.5, 2.5], terminal=[4.0, 2.5],
										dataBasePath='./pathplanning/11X11-AllCircle1/')
	num = 0
	while num < 30:
		# cap = cv.VideoWriter('test' +str(num)+'.mp4', cv.VideoWriter_fourcc('X', 'V', 'I', 'D'), 120.0, (env.width, env.height))
		# env.reset_random()
		# env.reset()
		env.reset_random_with_database()
		env.show_dynamic_image(isWait=True)
		while not env.is_terminal:
			# print(env.time)
			if cv.waitKey(1) == 27:
				return
			env.show_dynamic_image(isWait=False)
			wLeft = random.choice(env.action_space[0])
			wRight = random.choice(env.action_space[1])
			action = [wLeft, wRight]
			env.step_update(action=action)
			# print(env.current_state[0 : 4])
			if env.terminal_flag == 4:
				print(env.reward)
			# print(env.current _state)
		num += 1


def test_cartpole():
	from cartpole.cartpole import CartPole
	env = CartPole(initTheta=deg2rad(0), initX=0.0, save_cfg=True)
	num = 0
	while num < 10:
		env.reset()
		while not env.is_terminal:
			if cv.waitKey(1) == 27:
				return
			env.image = env.show.copy()
			env.show_dynamic_image(isWait=True)
			f = 0
			# f = random.uniform(-env.fm, env.fm)
			# f = -1 * math.sin(2 * math.pi * env.time)
			# print(env.x, env.dx, env.theta, env.dtheta)
			env.step_update(action=[f])
			print(env.time)
		num += 1


def test_cartpoleangleonly():
	from cartpole.cartpole_angleonly import CartPoleAngleOnly
	env = CartPoleAngleOnly(initTheta=deg2rad(10), save_cfg=True)
	num = 0
	while num < 1:
		env.reset()
		while not env.is_terminal:
			if cv.waitKey(1) == 27:
				return
			env.image = env.show.copy()
			env.show_dynamic_image(isWait=True)
			f = [5]
			# f = [random.uniform(-env.fm, env.fm)]
			# f = -1 * math.sin(2 * math.pi * env.time)
			# print(env.x, env.dx, env.theta, env.dtheta)
			env.step_update(action=f)
			cur_e_theta = rad2deg(env.current_state[0] / env.staticGain * env.thetaMax)
			nex_e_theta = rad2deg(env.next_state[0] / env.staticGain * env.thetaMax)
			print(cur_e_theta, nex_e_theta, env.reward)
			# print(env.time)
		num += 1


def test_cartpole_discrete_angleonly():
	from cartpole.cartpole_discrete_angle_only import CartPoleDiscreteAngleOnly
	env = CartPoleDiscreteAngleOnly(initTheta=deg2rad(10), save_cfg=True)
	num = 0
	i = 0
	while num < 5:
		env.reset()
		while not env.is_terminal:
			if cv.waitKey(1) == 27:
				return
			env.image = env.show.copy()
			# f = [env.action_space[0][i % env.action_num[0]]]
			f = [3]
			env.step_update(action=f)
			env.show_dynamic_image(isWait=False)
			i += 1
		num += 1


def test_uav_hover():
	from UAV.uav_hover import UAV_Hover
	import matplotlib.pyplot as plt
	env = UAV_Hover(target_pos=[0, 0, 4])
	env.uav_vis.arm_scale = 10  # 显示放大的尺度，自己设置即可
	num = 0
	plt.ion()
	while num < 2:
		env.reset_random()
		while not env.is_terminal:
			env.show_dynamic_image(per_show=1)
			# cv.waitKey(0)
			plt.pause(0.00000001)
			f = [random.uniform(env.fmin, env.fmax) for _ in range(4)]
			env.step_update(action=f)
		num += 1
	plt.ioff()


if __name__ == '__main__':
	# test_flight_attitude_simulator()
	# test_flight_attitude_simulator_continuous()
	# test_flight_attitude_simulator_2state_continuous()
	# test_ugv_bidirectional_continuous()
	# test_UGVBidirectional2()
	# test_SecondOrderIntegration()
	# test_ugv_forward_continuous()
	test_UGVForward()
	# test_ugv_forward_obstacles_continuous()
	# test_ugv_forward_discrete()
	# test_ugv_forward_obstacles_discrete()
	# test_cartpole()
	# test_cartpoleangleonly()
	# test_uav_hover()
	# test_cartpole_discrete_angleonly()
	pass
