import math

import numpy as np

from utils.functions import *
from algorithm.rl_base import rl_base
import cv2 as cv
from environment.color import Color


class CartPole(rl_base):
	def __init__(self, initTheta: float, initX: float):
		"""
		:param initTheta:       initial angle, which should be less than 30 degree
		:param initX:           initial position
		"""
		super(CartPole, self).__init__()
		'''physical parameters'''
		self.initTheta = initTheta
		self.initX = initX
		self.theta = self.initTheta
		self.x = self.initX
		self.dtheta = 0.  # 从左往右转为正
		self.dx = 0.  # 水平向左为正
		self.force = 0.  # 外力，水平向左为正

		self.theta_max = deg2rad(45)
		self.dtheta_max = deg2rad(90)
		self.x_max = 1.5
		self.dx_max = 3

		self.staticGain = 2.0

		self.M = 1.0  # mass of the cart
		self.m = 0.1  # mass of the pole
		self.g = 9.8
		self.ell = 0.2  # 1 / 2 length of the pole
		self.kf = 0.2  # friction coefficient
		self.fm = 8  # maximum force added on the cart

		self.dt = 0.02  # 10ms
		self.timeMax = 5  # maximum time of each episode
		self.time = 0.
		self.etheta = 0. - self.theta
		self.ex = 0. - self.x
		self.name = 'CartPole'
		'''physical parameters'''

		'''RL_BASE'''
		self.use_norm = True
		self.state_dim = 4  # theta, dtheta, x, dx
		self.state_num = [math.inf for _ in range(self.state_dim)]
		self.state_step = [None for _ in range(self.state_dim)]
		self.state_space = [None for _ in range(self.state_dim)]
		self.state_range = [[-self.staticGain, self.staticGain],
							[-math.inf, math.inf],
							[-self.staticGain, self.staticGain],
							[-math.inf, math.inf]]
		self.isStateContinuous = [True for _ in range(self.state_dim)]
		self.current_state = self.get_state()
		self.next_state = self.current_state.copy()

		self.action_dim = 1
		self.action_step = [None]
		self.action_range = [[-self.fm, self.fm]]
		self.action_num = [math.inf]
		self.action_space = [None]
		self.isActionContinuous = True
		self.initial_action = [self.force]
		self.current_action = self.initial_action.copy()

		self.reward = 0.0
		self.Q_x = 100  # cost for position error
		self.Q_dx = 0.2  # cost for linear velocity error
		self.Q_theta = 200  # cost for angular error
		self.Q_dtheta = 0.1  # cost for angular rate error
		self.R = 0.5  # cost for control input
		self.is_terminal = False
		self.terminal_flag = 0
		'''RL_BASE'''

		'''visualization_opencv'''
		self.width = 400
		self.height = 200
		self.image = np.ones([self.height, self.width, 3], np.uint8) * 255
		self.name4image = 'cartpole'
		self.xoffset = 0  # pixel
		self.scale = (self.width - 2 * self.xoffset) / 2 / self.x_max  # m -> pixel
		self.cart_x_pixel = 40  # 仅仅为了显示，比例尺不一样的
		self.cart_y_pixel = 30
		self.pixel_per_n = 20  # 每牛顿的长度
		self.pole_ell_pixel = 50
		self.image_copy = self.image.copy()
		# self.draw_slide()
		'''visualization_opencv'''

	def draw_slide(self):
		pt1 = (self.xoffset, int(self.height / 2) - 1)
		pt2 = (self.width - 1 - self.xoffset, int(self.height / 2) + 1)
		cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=-1)
		self.image_copy = self.image.copy()  # show是基础画布

	def draw_cartpole_force(self):
		# self.image = self.show.copy()
		cx = self.xoffset + (self.x + self.x_max) * self.scale
		cy = self.height / 2
		pt1 = (int(cx - self.cart_x_pixel / 2), int(cy + self.cart_y_pixel / 2))
		pt2 = (int(cx + self.cart_x_pixel / 2), int(cy - self.cart_y_pixel / 2))
		cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Orange, thickness=-1)

		pt1 = np.atleast_1d([int(cx), int(cy - self.cart_y_pixel / 2)])
		pt2 = np.atleast_1d([int(cx + self.pole_ell_pixel * math.sin(self.theta)),
							 int(cy - self.cart_y_pixel / 2 - self.pole_ell_pixel * math.cos(self.theta))])
		cv.line(img=self.image, pt1=pt1, pt2=pt2, color=Color().Red, thickness=4)
		if self.force >= 0:
			pt1 = np.atleast_1d([int(cx - self.cart_x_pixel / 2 - np.fabs(self.force) * self.pixel_per_n), int(cy)])
			pt2 = np.atleast_1d([int(cx - self.cart_x_pixel / 2), int(cy)])
		else:
			pt1 = np.atleast_1d([int(cx + self.cart_x_pixel / 2 + np.fabs(self.force) * self.pixel_per_n), int(cy)])
			pt2 = np.atleast_1d([int(cx + self.cart_x_pixel / 2), int(cy)])
		if np.fabs(self.force) > 1e-2:
			cv.circle(self.image, pt2, 5, Color().Red, -1)
			cv.line(self.image, pt1, pt2, Color().Red, 2, 8, 0)

	def draw_center(self):
		cv.circle(self.image, (int(self.xoffset + 1.5 * self.scale), int(self.height / 2)), 4, Color().Black, -1)

	def make_text(self):
		# self.image = self.show.copy()
		cv.putText(self.image, "time : %.2f s" % self.time, (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Black, 2)
		cv.putText(self.image, "theta: %.3f " % (rad2deg(self.theta)), (20, 40), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Black, 2)
		cv.putText(self.image, "  x  : %.3f m" % self.x, (20, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Black, 2)
		cv.putText(self.image, "force: %.3f N" % self.force, (20, 80), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Black, 2)

	def visualization(self):
		self.image = self.image_copy.copy()
		self.draw_cartpole_force()
		self.make_text()
		self.draw_center()
		cv.imshow(self.name4image, self.image)
		cv.waitKey(1)

	def get_state(self):
		if self.use_norm:
			s = np.array([self.theta / self.theta_max,
						  self.dtheta / self.dtheta_max,
						  self.x / self.x_max,
						  self.dx / self.dx_max]) * self.staticGain
		else:
			s = np.array([self.theta, self.dtheta, self.x, self.dx])
		return s

	def is_success(self):
		if np.linalg.norm([self.ex, self.dx, self.etheta, self.dtheta]) < 1e-2:
			return True
		return False

	def is_Terminal(self, param=None):
		"""
		:brief:     判断回合是否结束
		:return:    是否结束
		"""
		self.terminal_flag = 0
		self.is_terminal = False
		if (self.theta > self.theta_max + deg2rad(1)) or self.theta < -self.dtheta_max - deg2rad(1):
			self.terminal_flag = 1
			# print('Angle out...')
			self.is_terminal = True

		if self.x > self.x_max or self.x < -self.x_max:
			self.terminal_flag = 2
			# print('Position out...')
			self.is_terminal = True

		if self.time > self.timeMax:
			self.terminal_flag = 3
			# print('Time out')
			self.is_terminal = True

		if self.is_success():
			self.terminal_flag = 4
			# print('Success')
			self.is_terminal = True

	def get_reward(self, param=None):
		"""
		:param param:   extra parameters for reward function
		:return:
		"""
		Q_x = 5
		Q_dx = 0.0
		Q_theta = 1
		Q_omega = 0.0
		R = 0.01

		# theta_middle = self.theta_max / 2
		# x_middle = self.x_max / 2
		# r_x = (x_middle - np.fabs(self.x)) * Q_x
		# r_dx = -np.fabs(self.dx) * Q_dx
		# r_theta = (theta_middle - np.fabs(self.theta)) * Q_theta
		# r_omega = -np.fabs(self.dtheta) * Q_omega
		# r_f = -np.fabs(self.force) * R

		r_x = -np.fabs(self.x) * Q_x
		r_dx = -np.fabs(self.dx) * Q_dx
		r_theta = -np.fabs(self.theta) * Q_theta
		r_omega = -np.fabs(self.dtheta) * Q_omega
		r_f = -np.fabs(self.force) * R

		r_extra = 0.
		if self.terminal_flag == 1 or self.terminal_flag == 2:
			_n = (self.timeMax - self.time) / self.dt
			r_extra = _n * (r_x + r_dx + r_theta + r_omega + r_f)

		self.reward = r_x + r_dx + r_theta + r_omega + r_f + r_extra

	def ode(self, xx: np.ndarray):
		"""
		:param xx:  微分方程的状态，不是强化学习的状态。
		:return:
		"""
		'''微分方程里面的状态：[theta, dtheta, x, dx]'''
		_theta = xx[0]
		_dtheta = xx[1]
		_x = xx[2]
		_dx = xx[3]
		ddx = (self.force +
			   self.m * self.ell * _dtheta ** 2 * S(_theta)
			   - self.kf * _dx
			   - 3 / 4 * self.m * self.g * S(_theta) * C(_theta)) / \
			  (self.M + self.m - 3 / 4 * self.m * C(_theta) ** 2)
		ddtheta = 3 / 4 / self.m / self.ell * (self.m * self.g * S(_theta) - self.m * ddx * C(_theta))
		dx = _dx
		dtheta = _dtheta

		return np.array([dtheta, ddtheta, dx, ddx])

	def rk44(self, action: np.ndarray):
		[self.force] = action
		h = self.dt / 10  # RK-44 解算步长
		tt = self.time + self.dt
		xx = np.array([self.theta, self.dtheta, self.x, self.dx])
		while self.time < tt:
			temp = self.ode(xx)
			K1 = h * temp
			K2 = h * self.ode(xx + K1 / 2)
			K3 = h * self.ode(xx + K2 / 2)
			K4 = h * self.ode(xx + K3)
			xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
			self.time += h
		[self.theta, self.dtheta, self.x, self.dx] = xx.tolist()

	def step_update(self, action: list):
		self.force = action[0]  # get the extra force
		self.current_action = action.copy()
		self.current_state = self.get_state()
		self.rk44(np.array([self.force]))

		'''角度，位置误差更新'''
		self.etheta = 0. - self.theta
		self.ex = 0. - self.x
		self.is_Terminal()
		self.next_state = self.get_state()
		self.get_reward()

	def reset(self, random: bool = False):
		"""
		:brief:     reset
		:return:    None
		"""
		'''physical parameters'''
		if random:
			self.initTheta = np.random.uniform(-self.theta_max * 0.5, self.theta_max * 0.5)
			self.initX = np.random.uniform(-self.x_max / 2, self.x_max / 2)
		self.theta = self.initTheta
		self.x = self.initX
		self.dtheta = 0.  # 从左往右转为正
		self.dx = 0.  # 水平向左为正
		self.force = 0.  # 外力，水平向左为正
		self.time = 0.
		self.etheta = 0. - self.theta
		self.ex = 0. - self.x
		'''physical parameters'''

		'''RL_BASE'''
		self.current_state = self.get_state()
		self.next_state = self.current_state.copy()

		self.current_action = [0.]

		self.reward = 0.0
		self.is_terminal = False
		self.terminal_flag = 0
		'''RL_BASE'''
