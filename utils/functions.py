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


def point_is_in_poly(center, r, points: list, point: list) -> bool:
    """
    :brief:                     if a point is in a polygon
    :param center:              center of the circumcircle of the polygon
    :param r:                   radius of the circumcircle of the polygon
    :param points:              points of the polygon
    :param point:               the point to be tested
    :return:                    if the point is in the polygon
    """
    if center and r:
        if point_is_in_circle(center, r, point) is False:
            return False
    '''若在多边形对应的外接圆内，再进行下一步判断'''
    l_pts = len(points)
    res = False
    j = l_pts - 1
    for i in range(l_pts):
        if ((points[i][1] > point[1]) != (points[j][1] > point[1])) and \
                (point[0] < (points[j][0] - points[i][0]) * (point[1] - points[i][1]) / (
                        points[j][1] - points[i][1]) + points[i][0]):
            res = not res
        j = i
    if res is True:
        return True


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


def line_is_in_poly(center: list, r: float, points: list, point1: list, point2: list) -> bool:
    """
    :brief:             if a polygon and a line segment have an intersection
    :param center:      center of the circumcircle of the polygon
    :param r:           radius of the circumcircle of the polygon
    :param points:      points of the polygon
    :param point1:      the first point of the line segment
    :param point2:      the second point of the line segment
    :return:            if the polygon and the line segment have an intersection
    """
    if point_is_in_poly(center, r, points, point1):
        # print('Something wrong happened...')
        return True
    if point_is_in_poly(center, r, points, point2):
        # print('Something wrong happened...')
        return True
    length = len(points)
    for i in range(length):
        a = points[i % length]
        b = points[(i + 1) % length]
        c = point1.copy()
        d = point2.copy()
        '''通过坐标变换将a点变到原点'''
        b = [b[i] - a[i] for i in [0, 1]]
        c = [c[i] - a[i] for i in [0, 1]]
        d = [d[i] - a[i] for i in [0, 1]]
        a = [a[i] - a[i] for i in [0, 1]]
        '''通过坐标变换将a点变到原点'''

        '''通过坐标旋转将b点变到与X重合'''
        l_ab = dis_two_points(a, b)  # length of ab
        cos = b[0] / l_ab
        sin = b[1] / l_ab
        bb = [cos * b[0] + sin * b[1], -sin * b[0] + cos * b[1]]
        cc = [cos * c[0] + sin * c[1], -sin * c[0] + cos * c[1]]
        dd = [cos * d[0] + sin * d[1], -sin * d[0] + cos * d[1]]
        '''通过坐标旋转将b点变到与X重合'''

        if cc[1] * dd[1] > 0:
            '''如果变换后的cd纵坐标在x轴的同侧'''
            # return False
            continue
        else:
            '''如果变换后的cd纵坐标在x轴的异侧(包括X轴)'''
            if cc[0] == dd[0]:
                '''k == inf'''
                if min(bb) <= cc[0] <= max(bb):
                    return True
                else:
                    continue
            else:
                '''k != inf'''
                k_cd = (dd[1] - cc[1]) / (dd[0] - cc[0])
                b_cd = cc[1] - k_cd * cc[0]
                if k_cd != 0:
                    x_cross = -b_cd / k_cd
                    if min(bb) <= x_cross <= max(bb):
                        return True
                    else:
                        continue
                else:
                    '''k_cd == 0'''
                    if (min(bb) <= cc[0] <= max(bb)) or (min(bb) <= dd[0] <= max(bb)):
                        return True
                    else:
                        continue
    return False


def dis_point_2_poly(points, point):
    _l = len(points)
    dis = np.inf
    for i in range(_l):
        _dis = dis_point_2_line_segment(point, points[i], points[(i + 1) % _l])
        if _dis < dis:
            dis = _dis
    return dis


'''geometry'''
