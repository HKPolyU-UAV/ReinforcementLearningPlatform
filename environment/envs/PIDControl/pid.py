import math


class PID:
	def __init__(self, kp: float = 0, ki: float = 0., kd: float = 0.):
		self.kp = kp  # kp
		self.ki = ki  # ki
		self.kd = kd  # kd
		self.e = 0.  # error
		self.e_diff = 0.  # de/dt
		self.e_int = 0.  # integration of error
		self.p_out = 0.  # kp out
		self.i_out = 0.  # ki out
		self.d_out = 0.  # kd out

		self.dead_zone = 0.
		self.seg_linear = False  # 不考虑分段线性PID
		self.fuzzy = False  # 不考虑模糊PID
		self.neural = False  # 不考虑神经网络PID

	def set_pid(self, _kp, _ki, _kd):
		self.kp = _kp
		self.ki = _ki
		self.kd = _kd

	def set_multi_pid(self):
		'''pid 参数本身e, de ,\int{e} 的函数'''
		pass

	def set_e(self, _e):
		self.e_diff = _e - self.e
		self.e = _e
		self.e_int += _e

	def out(self):
		if math.fabs(self.e) >= self.dead_zone:
			self.p_out = self.kp * self.e
			self.i_out = self.ki * self.e_int
			self.d_out = self.kd * self.e_diff
		else:
			self.p_out = self.i_out = self.d_out = 0.
		return self.p_out + self.i_out + self.d_out

	def reset(self):
		self.__init__()
