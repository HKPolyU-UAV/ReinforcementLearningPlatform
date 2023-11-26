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


def sind(theta):
    return np.sin(theta / 180.0 * np.pi)


def cosd(theta):
    return np.cos(theta / 180.0 * np.pi)


'''geometry'''
def cal_vector_rad(v1: list, v2: list) -> float:
    """
    :brief:         calculate the rad between two vectors
    :param v1:      vector1
    :param v2:      vector2
    :return:        the rad
    """
    # print(v1, v2)
    if np.linalg.norm(v2) < 1e-4 or np.linalg.norm(v1) < 1e-4:
        return 0
    cosTheta = min(max(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1), 1)
    return np.arccos(cosTheta)


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


def cross_product(v1, v2) -> float:
    return v1[0] * v2[1] - v2[0] * v1[1]


def dis_two_points(pt1, pt2) -> float:
    return np.linalg.norm(pt1 - pt2)


def point_is_in_circle(c, r: float, pt) -> bool:
    return np.linalg.norm(c - pt) <= r


def point_is_in_ellipse(long, short, rotate_angle, c, pt) -> bool:
    trans = np.array([[cosd(-rotate_angle), -sind(-rotate_angle)], [sind(-rotate_angle), cosd(-rotate_angle)]])
    v = np.dot(trans, c - pt)
    return (v[0] / long) ** 2 + (v[1] / short) ** 2 <= 1


def dis_point_2_line_segment(pt: np.ndarray, pt1: np.ndarray, pt2: np.ndarray):
    ap = pt - pt1
    ab = pt2 - pt1
    ba = pt1 - pt2
    bp = pt - pt2
    cos_PAB = np.dot(ap, ab) / (np.linalg.norm(ap) * np.linalg.norm(ab))
    cos_PBA = np.dot(bp, ba) / (np.linalg.norm(bp) * np.linalg.norm(ba))
    if cos_PAB >= 0 and cos_PBA >= 0:
        return np.linalg.norm(ap) * np.sqrt(1 - cos_PAB ** 2)
    else:
        if cos_PAB < 0:
            return np.linalg.norm(ap)
        else:
            return np.linalg.norm(bp)


def line_is_in_ellipse(long, short, rotate_angle, c, p1, p2) -> bool:
    if point_is_in_ellipse(long, short, rotate_angle, c, p1) or point_is_in_ellipse(long, short, rotate_angle, c, p2):
        return True
    pt1 = p1 - c
    pt2 = p2 - c

    pptt1 = [pt1[0] * cosd(-rotate_angle) - pt1[1] * sind(-rotate_angle),
             pt1[0] * sind(-rotate_angle) + pt1[1] * cosd(-rotate_angle)]
    pptt2 = [pt2[0] * cosd(-rotate_angle) - pt2[1] * sind(-rotate_angle),
             pt2[0] * sind(-rotate_angle) + pt2[1] * cosd(-rotate_angle)]

    if pptt1[0] == pptt2[0]:
        if short ** 2 * (1 - pptt1[0] ** 2 / long ** 2) < 0:
            return False
        else:
            y_cross = np.sqrt(short ** 2 * (1 - pptt1[0] ** 2 / long ** 2))
            if max(pptt1[1], pptt2[1]) >= y_cross >= -y_cross >= min(pptt1[1], pptt2[1]):
                return True
            else:
                return False
    else:
        k = (pptt2[1] - pptt1[1]) / (pptt2[0] - pptt1[0])
        b = pptt1[1] - k * pptt1[0]
        ddelta = (long * short) ** 2 * (short ** 2 + long ** 2 * k ** 2 - b ** 2)
        if ddelta < 0:
            return False
        else:
            x_medium = -(k * b * long ** 2) / (short ** 2 + long ** 2 * k ** 2)
            if max(pptt1[0], pptt2[0]) >= x_medium >= min(pptt1[0], pptt2[0]):
                return True
            else:
                return False


def line_is_in_circle(c, r, pt1, pt2) -> bool:
    return line_is_in_ellipse(r, r, 0, c, pt1, pt2)


def dis_point_2_poly(points, point):
    _l = len(points)
    dis = np.inf
    for i in range(_l):
        _dis = dis_point_2_line_segment(point, points[i], points[(i + 1) % _l])
        if _dis < dis:
            dis = _dis
    return dis
'''geometry'''
