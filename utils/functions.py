import numpy as np


def deg2rad(deg):
    return deg * np.pi / 180.


def rad2deg(deg):
    return deg * 180. / np.pi


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def T(x):
    return np.tan(x)


def cal_vector_rad_oriented(v1, v2):
    """
    """
    '''有朝向的，从v1到v2'''
    if np.linalg.norm(v2) < 1e-4 or np.linalg.norm(v1) < 1e-4:
        return 0
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    return theta
