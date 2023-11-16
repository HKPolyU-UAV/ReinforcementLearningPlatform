import numpy as np


def ref_inner(time, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
    """
    :param time:        time
    :param amplitude:   amplitude
    :param period:      period
    :param bias_a:      amplitude bias
    :param bias_phase:  phase bias
    :return:            reference attitude angles and their 1st - 3rd derivatives
                        [phi_d, theta_d, psi_d]
                        [dot_phi_d, dot_theta_d, dot_psi_d]
                        [dot2_phi_d, dot2_theta_d, dot2_psi_d]
                        [dot3_phi_d, dot3_theta_d, dot3_psi_d]
    """
    w = 2 * np.pi / period
    _r = amplitude * np.sin(w * time + bias_phase) + bias_a
    _dr = amplitude * w * np.cos(w * time + bias_phase)
    _ddr = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
    _dddr = -amplitude * w ** 3 * np.cos(w * time + bias_phase)
    return _r, _dr, _ddr, _dddr


def ref_uav(time: float, amplitude: np.ndarray, period: np.ndarray, bias_a: np.ndarray, bias_phase: np.ndarray):
    """
    :param time:        time
    :param amplitude:   amplitude
    :param period:      period
    :param bias_a:      amplitude bias
    :param bias_phase:  phase bias
    :return:            reference position and yaw angle and their 1st - 3rd derivatives
                        [x_d, y_d, z_d, yaw_d]
                        [dot_x_d, dot_y_d, dot_z_d, dot_yaw_d]
                        [dot2_x_d, dot2_y_d, dot2_z_d, dot2_yaw_d]
                        [dot3_x_d, dot3_y_d, dot3_z_d, dot3_yaw_d]
    """
    w = 2 * np.pi / period
    _r = amplitude * np.sin(w * time + bias_phase) + bias_a
    _dr = amplitude * w * np.cos(w * time + bias_phase)
    _ddr = -amplitude * w ** 2 * np.sin(w * time + bias_phase)
    _dddr = -amplitude * w ** 3 * np.cos(w * time + bias_phase)
    return _r, _dr, _ddr, _dddr


def generate_uncertainty(time: float, is_ideal: bool = False) -> np.ndarray:
    """
    :param time:        time
    :param is_ideal:    ideal or not
    :return:            Fdx, Fdy, Fdz
    """
    if is_ideal:
        return np.array([0, 0, 0]).astype(float)
    else:
        T = 4
        w = 2 * np.pi / T
        Fdx = 0.8 * np.cos(w * time) + 0.6 * np.sin(w * time)
        Fdy = 0.8 * np.sin(w * time) + 0.6 * np.cos(w * time)
        Fdz = 0.8 * np.cos(w * time) + 0.6 * np.sin(w * time)

        return np.array([Fdx, Fdy, Fdz])
