import numpy as np
import cv2 as cv

from utils.functions import *
from algorithm.rl_base import rl_base
from environment.color import Color


class UGV(rl_base):
	def __init__(self,
				 pos0: np.ndarray = np.array([1., 1.]),
				 vel0: float = 0.,
				 phi0: float = 0.,
				 omega0: float = 0.,
				 map_size: np.ndarray = np.array([5.0, 5.0]),
				 target: np.ndarray = np.array([2.5, 2.5])):
		"""
		@param pos0:					initial position
		@param vel0:					initial velocity
		@param phi0:					initial heading angle
		@param omega0:					initial angular rate
		@param map_size:				map size
		@param target:					target
		@param is_controller_Bang3:		using Bang-3 control or not
		"""
		super(UGV, self).__init__()

		self.init_pos = pos0  # 初始位置
		self.init_vel = vel0  # 初始线速度
		self.init_phi = phi0  # 初始角度
		self.init_omega = omega0  # 初始角速度
		self.init_target = target  # 初始目标

		self.pos = pos0  # 位置
		self.vel = vel0  # 线速度
		self.phi = phi0  # 角度
		self.omega = omega0  # 角速度
		self.map_size = map_size  # 地图大小
		self.target = target  # 目标位置
		self.error = np.linalg.norm(self.target - self.pos)  # 位置误差

		'''hyper-parameters'''
		self.mass = 2.0  # 车体质量
		self.rBody = 0.25  # 车体半径
		self.l0 = 3 * self.rBody
		self.J = self.mass * self.rBody ** 2 / 2  # 车体转动惯量
		self.dt = 0.02  # 50Hz
		self.time = 0.  # time
		self.time_max = 6.0  # 每回合最大时间
		self.f = 0.  # 推力
		self.torque = 0.  # 转矩
		self.kf = 0.1  # 推力阻力系数
		self.kt = 0.1  # 转矩阻力系数
		'''hyper-parameters'''

		'''state limitation'''
		self.e_max = np.linalg.norm(self.map_size)
		self.v_max = 2
		self.phi_max = np.pi
		self.omega_max = 2 * np.pi
		'''state limitation'''

		self.miss = self.rBody  # 容许误差
		self.name = 'UGV'

		'''rl_base'''
		self.use_norm = True
		self.static_gain = 2
		self.state_dim = 4  # e, v, phi, dphi 位置误差，线速度，角度，角速度
		self.state_num = [np.inf for _ in range(self.state_dim)]
		self.state_step = [None for _ in range(self.state_dim)]
		self.state_space = [None for _ in range(self.state_dim)]
		self.isStateContinuous = [True for _ in range(self.state_dim)]
		self.state_range = np.array(
			[[0, self.e_max],
			 [-self.v_max, self.v_max],
			 [-self.phi_max, self.phi_max],
			 [-self.omega_max, self.omega_max]]
		)
		self.initial_state = self.get_state()
		self.current_state = self.initial_state.copy()
		self.next_state = self.initial_state.copy()

		self.action_dim = 2
		self.action_step = [None for _ in range(self.action_dim)]
		self.action_range = np.array([[self.fMin, self.fMax], [self.tMin, self.tMax]])
		self.action_num = [np.inf for _ in range(self.action_dim)]
		self.action_space = [None for _ in range(self.action_dim)]
		self.isActionContinuous = [True for _ in range(self.action_dim)]
		self.current_action = np.array([self.f, self.torque])

		self.reward = 0.0
		self.is_terminal = False
		self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
		'''rl_base'''

		'''visualization'''
		self.x_offset = 20
		self.y_offset = 20
		self.board = 170
		self.pixel_per_meter = 50
		self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array(
			[self.x_offset, self.y_offset])).astype(int)
		self.image_size[0] += self.board
		self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
		self.image_copy = self.image.copy()
		self.draw_init_image()
		'''visualization'''

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
		stepy = self.map_size / xNum
		for i in range(xNum):
			cv.line(self.image,
					self.dis2pixel([0, 0 + (i + 1) * stepy[1]]),
					self.dis2pixel([self.map_size[0], 0 + (i + 1) * stepy[1]]),
					Color().Black, 1)
		stepx = self.map_size / yNum
		for i in range(yNum):
			cv.line(self.image,
					self.dis2pixel([0 + (i + 1) * stepx, 0]),
					self.dis2pixel([0 + (i + 1) * stepx, self.map_size[1]]),
					Color().Black, 1)

	def draw_init_image(self):
		self.draw_boundary()
		self.draw_grid()
		self.image_copy = self.image.copy()

	def visualization(self):
		self.image = self.image_copy.copy()
		self.draw_car()
		self.draw_target()

		cv.putText(
			self.image,
			'time:   %.3fs' % (round(self.time, 3)),
			(self.image_size[0] - self.board - 5, 25), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

		cv.putText(
			self.image,
			'pos: [%.2f, %.2f]m' % (round(self.pos[0], 3), round(self.pos[1], 3)),
			(self.image_size[0] - self.board - 5, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
		cv.putText(
			self.image,
			'error: [%.2f, %.2f]m' % (round(self.error[0], 3), round(self.error[1], 3)),
			(self.image_size[0] - self.board - 5, 95), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
		cv.putText(
			self.image,
			'vel: %.2fm/s' % (round(self.vel, 2)),
			(self.image_size[0] - self.board - 5, 130), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
		cv.putText(
			self.image,
			'phi: %.2f ' % (round(rad2deg(self.phi), 2)),
			(self.image_size[0] - self.board - 5, 165), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
		cv.putText(
			self.image,
			'omega: %.2f PI' % (round(self.omega / np.pi, 2)),
			(self.image_size[0] - self.board - 5, 200), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

		cv.imshow(self.name, self.image)

	def draw_target(self):
		cv.circle(self.image, self.dis2pixel(self.target), 5, Color().random_color_by_BGR(), -1)

	def draw_car(self):
		pt1 = self.dis2pixel(self.pos)
		pt2 = self.dis2pixel(self.pos + np.array([self.l0 * np.cos(self.phi), self.l0 * np.sin(self.phi)]))
		cv.circle(self.image, pt1, self.length2pixel(self.rBody), Color().Orange, -1)  # 主体
		cv.line(self.image, pt1, pt2, Color().Blue, 2)
		cv.circle(self.image, pt2, 5, Color().Red, -1)  # 主体

	def get_state(self) -> np.ndarray:
		self.error = np.linalg.norm(self.target - self.pos)
		if self.use_norm:
			_s = 2 / self.e_max * self.error - 1
			_lv = self.vel / self.v_max
			_phi = self.phi / self.phi_max
			_omega = self.omega / self.omega_max
			return np.array([_s, _lv, _phi, _omega]) * self.static_gain
		else:
			return np.array([self.error, self.vel, self.phi, self.omega])

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
		b1 = np.linalg.norm(self.error) <= self.miss  # 位置误差小于 miss
		b2 = np.fabs(self.omega) < deg2rad(5)  # 角速度小于 5 度/秒
		b3 = np.linalg.norm(self.vel) < 0.02  # 线速度小于 0.02 m/s
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

	def get_reward(self, param=None):
		Q_pos = 1.
		Q_vel = 0.1
		Q_omega = 0.01

		u_pos = -self.error * Q_pos
		u_vel = -np.fabs(self.vel) * Q_vel
		u_omega = -np.fabs(self.omega) * Q_omega
		u_psi = 0.
		if self.terminal_flag == 1:		# 出界
			_n = (self.time_max - self.time) / self.dt
			u_psi = _n * (u_pos + u_vel + u_omega)

		self.reward = u_pos + u_vel + u_omega + u_psi

	def ode(self, xx: np.ndarray):
		"""
		@param xx:	state
		@return:	dx = f(x, t)，返回值当然是  dot{xx}
		"""
		[_x, _y, _phi, _omega, _vel] = xx[:]
		_dx = _vel * np.cos(_phi)
		_dy = _vel * np.sin(_phi)
		_dphi = _omega
		_domega = (self.torque - self.kt * _omega) / self.J
		_dvel = (self.f - self.kf * _vel) / self.mass
		return np.array([_dx, _dy, _dphi, _domega, _dvel])

	def rk44(self, action: np.ndarray):
		[self.f, self.torque] = action.copy()
		xx = np.array([self.pos[0], self.pos[1], self.phi, self.omega, self.vel])
		K1 = self.dt * self.ode(xx)
		K2 = self.dt * self.ode(xx + K1 / 2)
		K3 = self.dt * self.ode(xx + K2 / 2)
		K4 = self.dt * self.ode(xx + K3)
		xx = xx + (K1 + 2 * K2 + 2 * K3 + K4) / 6
		[self.pos[0], self.pos[1], self.phi, self.omega, self.vel] = xx[:]
		self.time += self.dt
		self.error = self.target - self.pos
		if self.phi > self.phi_max:
			self.phi -= 2 * self.phi_max
		if self.phi < -self.phi_max:
			self.phi += 2 * self.phi_max

	def step_update(self, action: np.ndarray):
		"""
		@param action:
		@return:
		"""
		self.current_action = action.copy()
		self.current_state = self.get_state()
		self.rk44(action=action)
		self.next_state = self.get_state()
		self.get_reward()

	def reset(self, random: bool = True):
		if random:
			d0 = 2 * self.rBody + 0.1
			self.init_pos = np.array([np.random.uniform(d0, self.map_size[0] - d0),
									  np.random.uniform(d0, self.map_size[1] - d0)])
			self.init_phi = np.random.uniform(-self.phi_max, self.phi_max)
			self.init_vel = np.random.uniform(-self.v_max, self.v_max)
			self.init_omega = 0.
		self.pos = self.init_pos.copy()
		self.vel = self.init_vel
		self.phi = self.init_phi
		self.omega = self.init_omega
		self.target = self.init_target.copy()
		self.error = self.target - self.pos
		self.time = 0.
		self.f = 0.
		self.torque = 0.

		self.initial_state = self.get_state()
		self.current_state = self.initial_state.copy()
		self.next_state = self.initial_state.copy()
		self.current_action = np.array([self.f, self.torque])
		self.reward = 0.0
		self.is_terminal = False
		self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功

		self.image = np.ones([self.image_size[1], self.image_size[0], 3], np.uint8) * 255
		self.image_copy = self.image.copy()
		self.draw_init_image()
