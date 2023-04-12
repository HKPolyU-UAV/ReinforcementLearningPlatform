import numpy as np

from common.common_func import *
from environment.envs import *


class UGV_Bidirectional(rl_base):
	def __init__(self,
				 pos0: np.ndarray = np.array([5.0, 5.0]),
				 vel0: float = 0.,
				 phi0: float = 0.,
				 omega0: float = 0.,
				 map_size: np.ndarray = np.array([10.0, 10.0]),
				 target: np.ndarray = np.array([5.0, 5.0]),
				 is_controller_BangBang: bool = False):
		"""
		@param pos0:					initial position
		@param vel0:					initial velocity
		@param phi0:					initial heading angle
		@param omega0:					initial angular rate
		@param map_size:				map size
		@param target:					target
		@param is_controller_Bang3:		using Bang-3 control or not
		"""
		super(UGV_Bidirectional, self).__init__()

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
		self.error = self.target - self.pos  # 位置误差

		'''hyper-parameters'''
		self.mass = 2.0  # 车体质量
		self.rBody = 0.25  # 车体半径
		self.J = self.mass * self.rBody ** 2 / 2  # 车体转动惯量
		self.dt = 0.02  # 50Hz
		self.time = 0.  # time
		self.timeMax = 6.0  # 每回合最大时间
		self.f = 0.  # 推力
		self.torque = 0.  # 转矩
		self.kf = 0.1  # 推力阻力系数
		self.kt = 0.3  # 转矩阻力系数
		'''hyper-parameters'''

		'''state limitation'''
		self.vMax = 2  # 最大线速度
		self.vMin = -2  # 最小线速度 (最大反向速度)
		self.fMax = 6  # 最大推力
		self.fMin = -6  # 最小推力 (最大拉力)
		self.tMax = np.pi / 4  # 最大转矩
		self.tMin = -np.pi / 4  # 最小转矩 (最大反向转矩)

		self.phiMax = np.pi  # 车体最大角度
		self.phiMin = -np.pi  # 车体最小角度
		self.omegaMax = 2 * np.pi  # 车体最大角速度
		self.omegaMin = -2 * np.pi  # 车体最小角速度
		'''state limitation'''

		self.miss = self.rBody  # 容许误差
		self.name = 'UGVBidirectional'

		'''rl_base'''
		self.is_controller_BangBang = is_controller_BangBang
		self.use_normalization = True
		self.static_gain = 2
		self.state_dim = 5  # ex, ey, v, phi, dphi 位置误差，线速度，角度，角速度
		self.state_num = [math.inf for _ in range(self.state_dim)]
		self.state_step = [None for _ in range(self.state_dim)]
		self.state_space = [None for _ in range(self.state_dim)]
		self.isStateContinuous = [True for _ in range(self.state_dim)]
		self.state_range = np.array(
			[[-self.map_size[0], self.map_size[0]],
			 [-self.map_size[1], self.map_size[1]],
			 [self.vMin, self.vMax],
			 [self.phiMin, self.phiMax],
			 [self.omegaMin, self.omegaMax]]
		)
		if self.use_normalization:
			self.initial_state = self.state_norm()
		else:
			self.initial_state = np.append(self.error, [self.vel, self.phi, self.omega])
		self.current_state = self.initial_state.copy()
		self.next_state = self.initial_state.copy()

		self.action_dim = 2
		if self.is_controller_BangBang:
			self.action_step = [None, None]		# 正常来说应该有数，但是这里不写也行，因为不会被调用
			self.action_range = [[self.fMin, self.fMax], [self.tMin, self.tMax]]
			self.action_num = [2, 2]
			self.action_space = [[self.fMin, self.fMax], [self.tMin, self.tMax]]
			self.isActionContinuous = [False, False]
			self.initial_action = np.array([self.f, self.torque])
			self.current_action = self.initial_action.copy()
		else:
			self.action_step = [None, None]
			self.action_range = [[self.fMin, self.fMax], [self.tMin, self.tMax]]
			self.action_num = [math.inf, math.inf]
			self.action_space = [None, None]
			self.isActionContinuous = [True, True]
			self.initial_action = np.array([self.f, self.torque])
			self.current_action = self.initial_action.copy()

		self.reward = 0.0
		self.is_terminal = False
		self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
		'''rl_base'''

		'''visualization'''
		self.x_offset = 20
		self.y_offset = 20
		self.board = 170
		self.pixel_per_meter = 50
		self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array([self.x_offset, self.y_offset])).astype(int)
		self.image_size[0] += self.board
		self.image = np.zeros([self.image_size[1], self.image_size[0], 3], np.uint8)
		self.image[:, :, 0] = np.ones([self.image_size[1], self.image_size[0]]) * 255
		self.image[:, :, 1] = np.ones([self.image_size[1], self.image_size[0]]) * 255
		self.image[:, :, 2] = np.ones([self.image_size[1], self.image_size[0]]) * 255
		self.image_white = self.image.copy()  # 纯白图
		'''visualization'''

		self.sum_d_theta = 0.

	def state_norm(self):
		state = np.append(self.error, [self.vel, self.phi, self.omega])
		norm_min = self.state_range[:, 0]
		norm_max = self.state_range[:, 1]
		norm_s = (2 * state - (norm_min + norm_max)) / (norm_max - norm_min) * self.static_gain
		return norm_s

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

	def show_dynamic_image(self, isWait=False):
		self.image = self.image_white.copy()
		self.draw_boundary()
		self.draw_car()
		self.draw_target()
		self.draw_grid()

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
		cv.waitKey(0) if isWait else cv.waitKey(1)

	def draw_car(self):
		cv.circle(self.image, self.dis2pixel(self.pos), self.length2pixel(self.rBody), Color().Orange, -1)  # 主体

		'''两个车轮'''
		# 额外画一个圆形，标志头
		line = points_move(points_rotate([[self.rBody, 0], [3 * self.rBody, 0]], self.phi), self.pos)
		cv.line(self.image, self.dis2pixel(line[0]), self.dis2pixel(line[1]), Color().Blue, 2)
		ball = points_move(points_rotate([3 * self.rBody, 0], self.phi), self.pos)
		cv.circle(self.image, self.dis2pixel(ball), self.length2pixel(0.08), Color().Red, -1)  # 主体

	def draw_boundary(self):
		cv.line(self.image, (self.x_offset, self.y_offset), (self.image_size[0] - self.x_offset - self.board, self.y_offset), Color().Black, 2)
		cv.line(self.image, (self.x_offset, self.y_offset), (self.x_offset, self.image_size[1] - self.y_offset), Color().Black, 2)
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

	def draw_target(self):
		cv.circle(self.image, self.dis2pixel(self.target), 5, Color().random_color_by_BGR(), -1)

	def draw_grid(self, num: np.ndarray = np.array([10, 10])):
		if np.min(num) <= 1:
			pass
		else:
			step = self.map_size / num
			for i in range(num[1] - 1):
				cv.line(self.image,
						self.dis2pixel([0, 0 + (i + 1) * step[1]]),
						self.dis2pixel([self.map_size[0], 0 + (i + 1) * step[1]]),
						Color().Black, 1)
			for i in range(num[0] - 1):
				cv.line(self.image,
						self.dis2pixel([0 + (i + 1) * step[0], 0]),
						self.dis2pixel([0 + (i + 1) * step[0], self.map_size[1]]),
						Color().Black, 1)

	def inverse_state_norm(self, s: np.ndarray):
		norm_min = self.state_range[:, 0]
		norm_max = self.state_range[:, 1]

		inverse_norm_s = (s * (norm_max - norm_min) / self.static_gain + (norm_min + norm_max)) / 2
		return inverse_norm_s

	def is_out(self):
		"""
		:return:
		"""
		'''简化处理，只判断中心的大圆有没有出界就好'''
		right_out = self.pos[0] + self.rBody > self.map_size[0]
		left_out = self.pos[0] - self.rBody < 0
		up_out = self.pos[1] + self.rBody > self.map_size[1]
		down_out = self.pos[1] - self.rBody < 0
		if right_out or left_out or up_out or down_out:
			return True
		return False

	def is_success(self):
		b1 = np.linalg.norm(self.error) <= self.miss  # 位置误差小于 miss
		b2 = np.fabs(self.omega) < deg2rad(5)  # 角速度小于 5 度/秒
		b3 = np.linalg.norm(self.vel) < 0.02  # 线速度小于 0.02 m/s
		return b1 and b2 and b3

	def is_Terminal(self, param=None):
		# if self.is_out():
		# 	print('...out...')
		# 	self.terminal_flag = 1
		# 	return True
		if self.time > self.timeMax:
			# print('...time out...')
			self.terminal_flag = 2
			return True
		if self.is_success():
			print('...success...')
			self.terminal_flag = 3
			return True
		self.terminal_flag = 0
		return False

	def get_reward(self, param=None):
		if self.use_normalization:
			cur_s = self.inverse_state_norm(self.current_state)
			nex_s = self.inverse_state_norm(self.next_state)
		else:
			cur_s = self.current_state
			nex_s = self.next_state
		# ex ey v phi omega
		cur_error = np.linalg.norm(cur_s[0: 2])
		nex_error = np.linalg.norm(nex_s[0: 2])

		cur_norm_error = cur_error / np.linalg.norm(self.map_size)		# 当前位置误差
		nex_norm_error = nex_error / np.linalg.norm(self.map_size)		# 下一时刻位置误差

		if self.is_controller_BangBang:
			r1 = -nex_norm_error - np.tanh(2.5 * nex_norm_error) + 1
			if self.time >= self.timeMax:		# 相当于时间最优之外， 加了一个终端约束，用于表征末状态的质量
				r2 = -nex_norm_error * 50
			else:
				r2 = 0
			if self.terminal_flag == 3:  # 成功
				r4 = 500
			elif self.terminal_flag == 2:  # 超时
				r4 = 0
			elif self.terminal_flag == 1:  # 出界
				r4 = -0
			else:
				r4 = 0

			self.reward = r1 + r2 + r4
		else:
			cur_phi = cur_s[-2]
			cur_v_head = np.array([np.cos(cur_phi), np.sin(cur_phi)])
			cur_error_theta = np.arccos(np.clip(np.dot(cur_s[0: 2], cur_v_head) / np.linalg.norm(cur_s[0: 2]), -1, 1))
			cur_error_theta = min(cur_error_theta, np.pi - cur_error_theta)
			cur_norm_error_theta = cur_error_theta / np.pi

			nex_phi = nex_s[-2]
			nex_v_head = np.array([np.cos(nex_phi), np.sin(nex_phi)])
			nex_error_theta = np.arccos(np.clip(np.dot(nex_s[0: 2], nex_v_head) / np.linalg.norm(nex_s[0: 2]), -1, 1))
			nex_error_theta = min(nex_error_theta, np.pi - nex_error_theta)
			nex_norm_error_theta = nex_error_theta / np.pi

			if self.sum_d_theta > 4 * np.pi:  # 如果转的超过两圈
				self.terminal_flag = 4
			# self.is_terminal = True

			'''4. 其他'''
			if self.terminal_flag == 4:  # 瞎几把转
				r4 = -0
			elif self.terminal_flag == 3:  # 成功
				r4 = 1000
			elif self.terminal_flag == 2:  # 超时
				r4 = -0
			elif self.terminal_flag == 1:  # 出界
				r4 = -0
			else:
				r4 = 0
			'''4. 其他'''
			# ex, ey, wl, wr, phi, dphi
			'''r1 是位置'''
			# if nex_norm_error >= 0.25:
			# 	kk = -180 * nex_norm_error + 45  # yyf_x0 = 0.25 时，kk = 0，误差大于0.25，开始罚，误差小于 0.25 开始奖励
			# 	r1 = nex_norm_error * kk  # nex_error
			# else:
			# 	kk = -180 * nex_norm_error + 45
			# 	r1 = (0.25 - nex_norm_error) * kk
			r1 = -(nex_norm_error * 2) ** 2

			'''r2 是角度'''
			if nex_error < 4 * self.miss:  # 如果误差比较小，就不考虑角度了
				r2 = 0
			else:
				r2 = -(nex_norm_error_theta * 1) ** 2
			r2 = 0
			self.reward = r1 + r2 + r4

	def ode(self, xx: np.ndarray):
		"""
		@param xx:	state
		@return:	dx = f(x, t)，返回值当然是  dot{xx}
		"""
		[_x, _y, _phi, _omega, _vel] = xx[:]
		# _phi = np.clip(_phi, self.phiMin, self.phiMax)			# 角度限制
		_dx = _vel * np.cos(_phi)
		_dy = _vel * np.sin(_phi)
		_dphi = _omega
		_domega = (self.torque - self.kt * _omega) / self.J
		_dvel = (self.f - self.kf * _vel) / self.mass
		return np.array([_dx, _dy, _dphi, _domega, _dvel])

	def rk44(self, action: np.ndarray):
		[self.f, self.torque] = action.copy()
		h = self.dt / 1
		tt = self.time + self.dt
		phi = self.phi
		while self.time < tt:
			xx_old = np.array([self.pos[0], self.pos[1], self.phi, self.omega, self.vel])
			K1 = h * self.ode(xx_old)
			K2 = h * self.ode(xx_old + K1 / 2)
			K3 = h * self.ode(xx_old + K2 / 2)
			K4 = h * self.ode(xx_old + K3)
			xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
			[self.pos[0], self.pos[1], self.phi, self.omega, self.vel] = xx_new.copy()
			self.omega = np.clip(self.omega, self.omegaMin, self.omegaMax)  # 角速度限制
			self.vel = np.clip(self.vel, self.vMin, self.vMax)  # 速度限制
			self.time += h
		self.error = self.target - self.pos
		self.sum_d_theta += np.fabs(phi - self.phi)
		if self.phi > self.phiMax:
			self.phi -= 2 * self.phiMax
		if self.phi < self.phiMin:
			self.phi -= 2 * self.phiMin

	def step_update(self, action: np.ndarray):
		"""
		@param action:
		@return:
		"""
		self.current_action = action.copy()
		if self.use_normalization:
			self.current_state = self.state_norm()
		else:
			self.current_state = np.append(self.error, [self.vel, self.phi, self.omega])

		'''rk44'''
		self.rk44(action=action)  # 在微分方程里的，不在里面的，都更新了，角度也有判断
		'''rk44'''

		self.is_terminal = self.is_Terminal()

		if self.use_normalization:
			self.next_state = self.state_norm()
		else:
			self.next_state = np.append(self.error, [self.vel, self.phi, self.omega])
		self.get_reward()

		'''其他'''

		'''其他'''

	def reset(self):
		self.pos = self.init_pos.copy()
		self.vel = self.init_vel
		self.phi = self.init_phi
		self.omega = self.init_omega
		self.target = self.init_target.copy()
		self.error = self.target - self.pos
		self.time = 0.
		self.sum_d_theta = 0.
		self.f = 0.
		self.torque = 0.

		if self.use_normalization:
			self.initial_state = self.state_norm()
		else:
			self.initial_state = np.append(self.error, [self.vel, self.phi, self.omega])
		self.current_state = self.initial_state.copy()
		self.next_state = self.initial_state.copy()
		self.initial_action = np.array([self.f, self.torque])
		self.current_action = self.initial_action.copy()
		self.reward = 0.0
		self.is_terminal = False
		self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功

	def reset_random(self):
		# self.init_pos = np.array([np.random.uniform(0 + self.rBody + 0.03, self.map_size[0] - self.rBody - 0.03),
		# 						  np.random.uniform(0 + self.rBody + 0.03, self.map_size[1] - self.rBody - 0.03)])
		self.init_pos = self.map_size / 2.0
		# self.init_phi = np.random.uniform(self.phiMin, self.phiMax)
		self.init_phi = 0.

		'''计算初始速度'''
		self.init_vel = np.random.uniform(self.vMin, self.vMax)

		self.init_omega = 0.
		self.init_target = np.array([np.random.uniform(0 + self.rBody + 0.1, self.map_size[0] - self.rBody - 0.1),
									 np.random.uniform(0 + self.rBody + 0.1, self.map_size[1] - self.rBody - 0.1)])

		self.pos = self.init_pos.copy()
		self.vel = self.init_vel
		self.phi = self.init_phi
		self.omega = self.init_omega
		self.target = self.init_target.copy()
		self.error = self.target - self.pos
		self.time = 0.
		self.sum_d_theta = 0.

		if self.use_normalization:
			self.initial_state = self.state_norm()
		else:
			self.initial_state = np.append(self.error, [self.vel, self.phi, self.omega])
		self.current_state = self.initial_state.copy()
		self.next_state = self.initial_state.copy()
		self.initial_action = np.array([self.f, self.torque])
		self.current_action = self.initial_action.copy()
		self.reward = 0.0
		self.is_terminal = False
		self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
