import numpy as np


class fntsmc_param:
    def __init__(self):
        self.k1: np.ndarray = np.array([1.2, 0.8, 1.5])
        self.k2: np.ndarray = np.array([0.2, 0.6, 1.5])
        self.alpha: np.ndarray = np.array([1.2, 1.5, 1.2])
        self.beta: np.ndarray = np.array([0.3, 0.3, 0.3])
        self.gamma: np.ndarray = np.array([0.2, 0.2, 0.2])
        self.lmd: np.ndarray = np.array([2.0, 2.0, 2.0])
        self.dim: int = 3
        self.dt: float = 0.01
        self.ctrl0: np.ndarray = np.array([0., 0., 0.])
        self.saturation: np.ndarray = np.array([0., 0., 0.])


class fntsmc_pos:
    def __init__(self, param: fntsmc_param):
        self.k1 = param.k1
        self.k2 = param.k2
        self.alpha = param.alpha
        self.beta = param.beta
        self.gamma = param.gamma
        self.lmd = param.lmd
        self.dt = param.dt
        self.dim = param.dim

        self.sigma_o = np.zeros(self.dim)
        self.dot_sigma_o1 = np.zeros(self.dim)
        self.sigma_o1 = np.zeros(self.dim)
        self.so = self.sigma_o + self.lmd * self.sigma_o1
        # self.ctrl = np.zeros(self.dim)
        self.control = param.ctrl0

    def control_update(self, kp: float, m: float, vel: np.ndarray, e: np.ndarray, de: np.ndarray, dd_ref: np.ndarray, obs: np.ndarray):
        """
        :param kp:
        :param m:
        :param vel:
        :param e:
        :param de:
        :param dd_ref:
        :param obs:
        :brief:         输出为 x y z 三轴的虚拟的加速度
        :return:
        """
        k_tanh_e = 5
        k_tanh_sigma0 = 5
        self.sigma_o = de + self.k1 * e + self.gamma * np.fabs(e) ** self.alpha * np.tanh(k_tanh_e * e)
        self.dot_sigma_o1 = np.fabs(self.sigma_o) ** self.beta * np.tanh(k_tanh_sigma0 * self.sigma_o)
        self.sigma_o1 += self.dot_sigma_o1 * self.dt
        self.so = self.sigma_o + self.lmd * self.sigma_o1

        uo1 = kp / m * vel + dd_ref - self.k1 * de - self.gamma * self.alpha * np.fabs(e) ** (self.alpha - 1) * de - self.lmd * self.dot_sigma_o1
        uo2 = -self.k2 * self.so - obs

        self.control = uo1 + uo2


class fntsmc_att:
    def __init__(self, param: fntsmc_param):
        self.k1 = param.k1
        self.k2 = param.k2
        self.alpha = param.alpha
        self.beta = param.beta
        self.gamma = param.gamma
        self.lmd = param.lmd
        self.dt = param.dt

        self.dim = param.dim

        self.s = np.zeros(self.dim)
        self.dot_s1 = np.zeros(self.dim)
        self.s1 = np.zeros(self.dim)
        self.sigma = self.s + self.lmd * self.s1
        self.control = param.ctrl0
        self.saturation = param.saturation

    def control_update(self,
                       second_order_att_dynamics: np.ndarray,
                       control_mat: np.ndarray,
                       e: np.ndarray,
                       de: np.ndarray,
                       dd_ref: np.ndarray):
        """
        @param second_order_att_dynamics:
        @param control_mat:
        @param e:
        @param de:
        @return:
        """
        k_tanh_e = 5
        k_tanh_s = 5
        k_tanh_sigma = 10
        self.s = 1 * de + self.k1 * e + self.gamma * np.fabs(e) ** self.alpha * np.tanh(k_tanh_e * e)
        self.dot_s1 = np.fabs(self.s) ** self.beta * np.tanh(k_tanh_s * self.s)
        self.s1 += self.dot_s1 * self.dt
        self.sigma = self.s + self.lmd * self.s1

        u1 = second_order_att_dynamics + dd_ref + self.k1 * de + self.gamma * self.alpha * np.fabs(e) ** (self.alpha - 1) * de + self.lmd * self.dot_s1
        u2 = -self.k2 * np.tanh(k_tanh_sigma * self.sigma)
        # u2 = -self.k2 * self.sigma

        self.control = -np.dot(np.linalg.inv(control_mat), u1 + u2)
        self.control = np.clip(self.control, -self.saturation, self.saturation)
