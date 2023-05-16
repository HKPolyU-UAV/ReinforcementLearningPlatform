import math
import random
import re
import numpy as np
from numpy import linalg
import scipy.spatial as spt


def deg2rad(deg: float) -> float:
    """
    :brief:         omit
    :param deg:     degree
    :return:        radian
    """
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    :brief:         omit
    :param rad:     radian
    :return:        degree
    """
    return rad * 180.8 / math.pi


def str2list(string: str) -> list:
    """
    :brief:         transfer a string to list，必须是具备特定格式的：内外两层都是list
    :param string:  string
    :return:        the list
    """
    res = re.split(r'[\[\]]', string.strip())
    inner = []
    outer = []
    for item in res:
        item.strip()
    while '' in res:
        res.remove('')
    while ', ' in res:
        res.remove(', ')
    while ',' in res:
        res.remove(',')
    while ' ' in res:
        res.remove(' ')
    for _res in res:
        _res_spilt = re.split(r',', _res)
        for item in _res_spilt:
            inner.append(float(item))
        outer.append(inner.copy())
        inner.clear()
    return outer


def str2list2(string: str) -> list:
    """
    @note:          transfer a string to list，必须是具备特定格式的：外层list，内层int
    @param string:  string
    @return:        the list
    """
    res = re.split(r'[\[\]]', string.strip())
    outer = []
    for item in res:
        item.strip()
    while '' in res:
        res.remove('')
    while ', ' in res:
        res.remove(', ')
    while ',' in res:
        res.remove(',')
    while ' ' in res:
        res.remove(' ')
    for _res in res:
        outer.append(int(_res))
    return outer


def sind(theta: float) -> float:
    """
    :param theta:   degree, not rad
    :return:
    """
    return math.sin(theta / 180.0 * math.pi)


def cosd(theta: float) -> float:
    """
    :param theta:   degree, not rad
    :return:
    """
    return math.cos(theta / 180.0 * math.pi)


def points_rotate(pts: list, theta: float) -> list:
    """
    :param pts:
    :param theta:   rad, counter-clockwise
    :return:        new position
    """
    if type(pts[0]) == list:
        return [[math.cos(theta) * pt[0] - math.sin(theta) * pt[1], math.sin(theta) * pt[0] + math.cos(theta) * pt[1]] for pt in pts]
    else:
        return [math.cos(theta) * pts[0] - math.sin(theta) * pts[1], math.sin(theta) * pts[0] + math.cos(theta) * pts[1]]


def points_move(pts: list, dis) -> list:
    if type(pts[0]) == list:
        return [[pt[0] + dis[0], pt[1] + dis[1]] for pt in pts]
    else:
        return [pts[0] + dis[0], pts[1] + dis[1]]


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
    return math.acos(cosTheta)


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


def cross_product(vec1: list, vec2: list) -> float:
    """
    :brief:         cross product of two vectors
    :param vec1:    vector1
    :param vec2:    vector2
    :return:        cross product
    """
    return vec1[0] * vec2[1] - vec2[0] * vec1[1]


def dis_two_points(point1: list, point2: list) -> float:
    """
    :brief:         euclidean distance between two points
    :param point1:  point1
    :param point2:  point2
    :return:        euclidean distance
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def point_is_in_circle(center: list, r: float, point: list) -> bool:
    """
    :brief:         if a point is in a circle
    :param center:  center of the circle
    :param r:       radius of the circle
    :param point:   point
    :return:        if the point is in the circle
    """
    sub = [center[i] - point[i] for i in [0, 1]]
    return np.linalg.norm(sub) <= r


def point_is_in_ellipse(long: float, short: float, rotate_angle: float, center: list, point: list) -> bool:
    """
    :brief:                     判断点是否在椭圆内部
    :param long:                长轴
    :param short:               短轴
    :param rotate_angle:        椭圆自身的旋转角度
    :param center:              中心点
    :param point:               待测点
    :return:                    bool
    """
    sub = np.array([point[i] - center[i] for i in [0, 1]])
    trans = np.array([[cosd(-rotate_angle), -sind(-rotate_angle)], [sind(-rotate_angle), cosd(-rotate_angle)]])
    [x, y] = list(np.dot(trans, sub))
    return (x / long) ** 2 + (y / short) ** 2 <= 1


def dis_point_2_line_segment(pt, pt1, pt2):
    p = np.array(pt)
    a = np.array(pt1)
    b = np.array(pt2)
    ap = p - a
    ab = b - a
    ba = a - b
    bp = p - b
    cos_PAB = np.dot(ap, ab) / (np.linalg.norm(ap) * np.linalg.norm(ab))
    cos_PBA = np.dot(bp, ba) / (np.linalg.norm(bp) * np.linalg.norm(ba))
    if cos_PAB >= 0 and cos_PBA >= 0:
        return np.linalg.norm(ap) * np.sqrt(1 - cos_PAB ** 2)
    else:
        if cos_PAB < 0:
            return np.linalg.norm(ap)
        else:
            return np.linalg.norm(bp)


def cross_2_line_seg(seg11, seg12, seg21, seg22):
    a = np.array(seg11)
    b = np.array(seg12)
    c = np.array(seg21)
    d = np.array(seg22)

    if (max(c[0], d[0]) < min(a[0], b[0])) or (max(c[1], d[1]) < min(a[1], b[1])) or (max(a[0], b[0]) < min(c[0], d[0])) or (max(a[1], b[1]) < min(c[1], d[1])):
        return False, None

    else:  # 一定有交点
        if a[0] == b[0]:
            kk = (c[1] - d[1]) / (c[0] - d[0])
            bb = c[1] - kk * c[0]
            y = kk * a[0] + bb
            if (y > max(a[1], b[1])) or (y < min(a[1], b[1])):
                return False, None
            else:
                return True, [a[0], y]
        elif c[0] == d[0]:
            kk = (a[1] - b[1]) / (a[0] - b[0])
            bb = a[1] - kk * a[0]
            y = kk * c[0] + bb
            if (y > max(c[1], d[1])) or (y < min(c[1], d[1])):
                return False, None
            else:
                return True, [c[0], y]
        else:
            kk1 = (a[1] - b[1]) / (a[0] - b[0])
            bb1 = a[1] - kk1 * a[0]
            kk2 = (c[1] - d[1]) / (c[0] - d[0])
            bb2 = c[1] - kk2 * c[0]
            x = (bb1 - bb2) / (kk2 - kk1)
            y = kk1 * x + bb1
            if (x > min(max(a[0], b[0]), max(c[0], d[0]))) or (x < max(min(a[0], b[0]), min(c[0], d[0]))) or \
                    (y > min(max(a[1], b[1]), max(c[1], d[1]))) or (y < max(min(a[1], b[1]), min(c[1], d[1]))):
                return False, None
            else:
                return True, [x, y]


def cross_pt_ray_2_poly(ray_s, ray_t, points):
    _l = len(points)
    PT = []
    dis = np.inf
    HAVE = False
    for i in range(_l):
        have, pt = cross_2_line_seg(ray_s, ray_t, points[i], points[(i + 1) % _l])
        if have:
            _dis = dis_two_points(pt, ray_s)
            HAVE = True
            if _dis < dis:
                dis = _dis
                PT = pt.copy()
    return HAVE, PT, dis


def dis_point_2_poly(points, point):
    _l = len(points)
    dis = np.inf
    for i in range(_l):
        _dis = dis_point_2_line_segment(point, points[i], points[(i + 1) % _l])
        if _dis < dis:
            dis = _dis
    return dis


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


def line_is_in_ellipse(long: float, short: float, rotate_angle: float, center: list, point1: list,
                       point2: list) -> bool:
    """
    :brief:                     判断线段与椭圆是否有交点
    :param long:                长轴
    :param short:               短轴
    :param rotate_angle:        椭圆自身的旋转角度
    :param center:              中心点
    :param point1:              待测点1
    :param point2:              待测点2
    :return:                    bool
    """
    if point_is_in_ellipse(long, short, rotate_angle, center, point1):
        return True
    if point_is_in_ellipse(long, short, rotate_angle, center, point2):
        return True
    pt1 = [point1[i] - center[i] for i in [0, 1]]
    pt2 = [point2[j] - center[j] for j in [0, 1]]  # 平移至原点

    pptt1 = [pt1[0] * cosd(-rotate_angle) - pt1[1] * sind(-rotate_angle),
             pt1[0] * sind(-rotate_angle) + pt1[1] * cosd(-rotate_angle)]
    pptt2 = [pt2[0] * cosd(-rotate_angle) - pt2[1] * sind(-rotate_angle),
             pt2[0] * sind(-rotate_angle) + pt2[1] * cosd(-rotate_angle)]

    if pptt1[0] == pptt2[0]:
        if short ** 2 * (1 - pptt1[0] ** 2 / long ** 2) < 0:
            return False
        else:
            y_cross = math.sqrt(short ** 2 * (1 - pptt1[0] ** 2 / long ** 2))
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


def line_is_in_circle(center: list, r: float, point1: list, point2: list) -> bool:
    """
    :brief:             if a circle and a line segment have an intersection
    :param center:      center of the circle
    :param r:           radius of the circle
    :param point1:      point1 of the line segment
    :param point2:      point2 of the line segment
    :return:            if the circle and the line segment have an intersection
    """
    return line_is_in_ellipse(r, r, 0, center, point1, point2)


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


def uniform_sample_on_ellipse(center, long, short, theta, numPt):
    pts = []
    for _ in range(numPt):
        a = random.random() * 2 * math.pi
        ra = long
        rb = short
        x = center[0] + ra * math.cos(a) * math.cos(theta) - rb * math.sin(a) * math.sin(theta)
        y = center[1] + rb * math.sin(a) * math.cos(theta) + ra * math.cos(a) * math.sin(theta)
        pts.append([x, y])
    return pts


def uniform_sample_in_ellipse(center, long, short, theta, numPt):
    pts = []
    for _ in range(numPt):
        a = random.random() * 2 * math.pi
        ra = random.random() ** 0.5 * long
        rb = random.random() ** 0.5 * short
        x = center[0] + ra * math.cos(a) * math.cos(theta) - rb * math.sin(a) * math.sin(theta)
        y = center[1] + rb * math.sin(a) * math.cos(theta) + ra * math.cos(a) * math.sin(theta)
        pts.append([x, y])
    return pts


def uniform_sample_in_circle(center, r, numPt):
    pts = []
    for _ in range(numPt):
        a = random.random() * 2 * math.pi
        r = random.uniform(0, r)
        x = center[0] + r * math.cos(a)
        y = center[1] + r * math.sin(a)
        pts.append([x, y])
    return pts


def cal_two_pt_set_dis(ptSet1, ptSet2):
    res = [None, None]
    dis = np.inf
    for pt1 in ptSet1:
        for pt2 in ptSet2:
            new_dis = dis_two_points(pt1, pt2)
            if dis > new_dis:
                dis = new_dis
                res = [pt1, pt2]
    return dis, res


def find_convex_hull(nodes):
    p = np.array(nodes)
    hull = spt.ConvexHull(p)
    pt = []
    for i in hull.vertices:
        pt.append(list(nodes[i]))
    return pt


def is_point_in_convex_hull(nodes, point):
    lll = len(nodes)
    s1 = cross_product([point[j] - nodes[0][j] for j in [0, 1]], [nodes[(j + 1) % lll][j] - nodes[0][j] for j in [0, 1]])
    for i in range(lll):
        vec1 = [point[j] - nodes[i][j] for j in [0, 1]]
        vec2 = [nodes[(i + 1) % lll][j] - nodes[i][j] for j in [0, 1]]
        s2 = cross_product(vec1, vec2)
        if s1 * s2 < 0:
            return False
        else:
            s1 = s2
    return True


def get_convex_hull_area(nodes):
    start = nodes[0]
    area = 0
    for i in range(len(nodes) - 2):
        s1 = start
        s2 = nodes[i + 1]
        s3 = nodes[i + 2]
        x1 = s2[0] - s1[0]
        y1 = s2[1] - s1[1]
        x2 = s3[0] - s1[0]
        y2 = s3[1] - s1[1]
        area += 0.5 * np.fabs(x1 * y2 - x2 * y1)
    return area


def convex_hull_dilate(nodes, d):
    center = [0, 0]
    new = []
    for i in range(len(nodes)):
        center[0] += nodes[i][0]
        center[1] += nodes[i][1]
    center = [center[i] / len(nodes) for i in [0, 1]]
    for i in range(len(nodes)):
        s = dis_two_points(center, nodes[i]) / d
        x = nodes[i][0] + (nodes[i][0] - center[0]) * s
        y = nodes[i][1] + (nodes[i][1] - center[1]) * s
        new.append([x, y])
    return new, center


def convex_hull_dilate2(nodes, d):
    ll = len(nodes)
    center = [0, 0]
    line = []
    new_line = []
    if ll > 2:
        '''1. 得到原来的直线'''
        for i in range(ll):
            [x1, y1] = nodes[i]
            [x2, y2] = nodes[(i + 1) % ll]
            center[0] += x1
            center[1] += y1
            if x1 == x2:
                line.append([np.inf, x1])
            else:
                k = (y1 - y2) / (x1 - x2)
                b = y1 - k * x1
                line.append([k, b])
        '''对于夹角过于小的点，做直线填充'''
        insert_node = []  # 新插入的点编号
        insert_line = []  # 新插入的直线
        for i in range(len(line)):  # 第i个角
            pt1 = nodes[i - 1]
            pt2 = nodes[i]
            pt3 = nodes[(i + 1) % len(line)]
            vec1 = [pt3[j] - pt2[j] for j in [0, 1]]
            vec2 = [pt1[j] - pt2[j] for j in [0, 1]]
            theta = cal_vector_rad_oriented(vec1, vec2)
            '''凸包的点都是逆时针的'''
            if theta <= np.fabs(deg2rad(62)):  # 如果夹角比较小，那么做直线填充，并同样扩充d，这个夹角是从vec1旋转到vec2的，顺序别弄错了
                insert_node.append(i)
                new_theta = cal_vector_rad_oriented([1, 0], vec1) + theta / 2  # 新的直线斜率
                k = np.inf if new_theta == np.pi / 2 else np.tan(new_theta)
                k = 0 if k == np.inf else -1 / k
                b = pt2[1] - k * pt2[0]
                insert_line.append([k, b])
            else:
                pass
        if insert_node:
            insert_node.reverse()
            insert_line.reverse()
            for j in range(len(insert_line)):
                line.insert(insert_node[j], insert_line[j])
        center = [center[i] / ll for i in [0, 1]]  # 得到中心点位置，如果增加过线，那么中心点位置就会变，不过不用管，不会影响最终结果
        [x0, y0] = center

        '''2. 计算扩充的直线'''
        for _line in line:
            [k, b] = _line
            if k == np.inf:
                b1 = b + d if x0 < b else b - d
                new_line.append([k, b1])
            else:
                d0 = np.fabs(k * x0 - y0 + b) / np.sqrt(k ** 2 + 1)
                d1 = np.fabs(k * x0 - y0 + b + d * np.sqrt(k ** 2 + 1)) / np.sqrt(k ** 2 + 1)
                if d1 > d0 and d1 > d:
                    b1 = b + d * np.sqrt(k ** 2 + 1)
                else:
                    b1 = b - d * np.sqrt(k ** 2 + 1)
                new_line.append([k, b1])

    elif ll == 2:
        '''原来的凸包就是一条直线'''
        [x1, y1] = nodes[0]
        [x2, y2] = nodes[1]
        # x0 = (x1 + x2) / 2
        # y0 = (y1 + y2) / 2
        if x1 == x2:
            new_line.append([0, min(y1, y2) - d])
            new_line.append([np.inf, x1 + d])
            new_line.append([0, max(y1, y2) + d])
            new_line.append([np.inf, x1 - d])
        elif y1 == y2:
            new_line.append([0, y1 - d])
            new_line.append([np.inf, max(x1, x2) + d])
            new_line.append([0, y1 + d])
            new_line.append([np.inf, min(x1, x2) - d])
        else:
            k = (y1 - y2) / (x1 - x2)
            b = y1 - k * x1
            if k > 0:
                if x1 > x2:
                    new_line.append([k, b - d * np.sqrt(k ** 2 + 1)])
                    new_line.append([-1 / k, y1 + x1 / k + d * np.sqrt(1 / k ** 2 + 1)])
                    new_line.append([k, b + d * np.sqrt(k ** 2 + 1)])
                    new_line.append([-1 / k, y2 + x2 / k - d * np.sqrt(1 / k ** 2 + 1)])
                else:
                    new_line.append([k, b - d * np.sqrt(k ** 2 + 1)])
                    new_line.append([-1 / k, y2 + x2 / k + d * np.sqrt(1 / k ** 2 + 1)])
                    new_line.append([k, b + d * np.sqrt(k ** 2 + 1)])
                    new_line.append([-1 / k, y1 + x1 / k - d * np.sqrt(1 / k ** 2 + 1)])
            else:
                if x1 > x2:
                    new_line.append([-1 / k, y1 + x1 / k - d * np.sqrt(1 / k ** 2 + 1)])
                    new_line.append([k, b + d * np.sqrt(k ** 2 + 1)])
                    new_line.append([-1 / k, y2 + x2 / k + d * np.sqrt(1 / k ** 2 + 1)])
                    new_line.append([k, b - d * np.sqrt(k ** 2 + 1)])
                else:
                    new_line.append([-1 / k, y2 + x2 / k - d * np.sqrt(1 / k ** 2 + 1)])
                    new_line.append([k, b + d * np.sqrt(k ** 2 + 1)])
                    new_line.append([-1 / k, y1 + x1 / k + d * np.sqrt(1 / k ** 2 + 1)])
                    new_line.append([k, b - d * np.sqrt(k ** 2 + 1)])

    else:
        [x0, y0] = nodes[0]
        new_line.append([0, y0 - d])
        new_line.append([np.inf, x0 + d])
        new_line.append([0, y0 + d])
        new_line.append([np.inf, x0 - d])

    '''计算新的交点'''
    new_nodes = []
    for i in range(len(new_line)):
        [k1, b1] = new_line[i - 1]
        [k2, b2] = new_line[i]
        if k1 == k2:
            print('in function \' convex_hull_dilate2 \'error...')
            exit(0)
        if k1 == np.inf:
            x = b1
            y = k2 * x + b2
        elif k2 == np.inf:
            x = b2
            y = k1 * x + b1
        else:
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
        new_nodes.append([x, y])
    # return new_nodes, center
    return new_nodes


def get_minAeraEllipse(nodes=None, tolerance=0.01):
    P = np.array(nodes)
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT, np.dot(linalg.inv(V), Q)))  # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
        np.dot(P.T, np.dot(np.diag(u), P)) -
        np.array([[a * b for b in center] for a in center])
    ) / d

    # Get the values we'd like to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0 / np.sqrt(s)

    # return center, radii, rotation, math.acos(rotation[0][0])
    return center, radii[0], radii[1], math.acos(rotation[0][0]) * np.sign(rotation[0][1])


def uniform_sample_in_triangle(nodes, num):
    r1 = np.sqrt(np.random.uniform(0, 1, num))
    r2 = np.random.uniform(0, 1, num)
    array = np.array(nodes)
    ab = array[1] - array[0]
    bc = array[2] - array[1]
    ap = []
    for i in range(num):
        n = r1[i] * ab + r1[i] * r2[i] * bc + array[0]
        ap.append(list(n))
    return ap


def get_convex_hull_triangle(hull):
    p1 = hull[0]
    ss = 0
    t = []
    for i in range(len(hull) - 2):
        p2 = hull[(i + 1) % len(hull)]
        p3 = hull[(i + 2) % len(hull)]
        s = np.fabs(cross_product(vec1=[p2[0] - p1[0], p2[1] - p1[1]], vec2=[p3[0] - p1[0], p3[1] - p1[1]])) * 0.5
        ss += s
        t.append([p1, p2, p3, s])
    for i in range(len(t)):
        t[i][3] /= ss  # 得到面积的比例
    return t


def uniform_sample_in_hull(hull, num):
    triangle = get_convex_hull_triangle(hull)
    iPDF = []
    for i in range(len(triangle)):
        if i == 0:
            iPDF.append(triangle[i][3])
        else:
            iPDF.append(iPDF[i - 1] + triangle[i][3])

    pts = []
    for _ in range(num):
        index = np.random.uniform(0, 1, 1)
        for i in range(len(iPDF)):
            if index <= iPDF[i]:
                index = i
                break
        pt = uniform_sample_in_triangle(triangle[index][0:3], 1)[0]
        pts.append(pt)

    return pts


def uniform_sample_between_two_hull(inner, outer, num):
    assert len(inner) == len(outer)
    t = []
    ll = len(inner)
    ss = 0
    for i in range(ll):
        p1 = inner[i]
        p2 = outer[i]
        p3 = inner[(i + 1) % ll]
        p4 = outer[(i + 1) % ll]
        s1 = np.fabs(cross_product(vec1=[p2[0] - p1[0], p2[1] - p1[1]], vec2=[p4[0] - p1[0], p4[1] - p1[1]])) * 0.5  # 124
        s2 = np.fabs(cross_product(vec1=[p3[0] - p1[0], p3[1] - p1[1]], vec2=[p4[0] - p1[0], p4[1] - p1[1]])) * 0.5  # 134
        ss += s1 + s2
        t.append([p1, p2, p4, s1])
        t.append([p1, p3, p4, s2])
    for i in range(len(t)):
        t[i][3] /= ss

    '''得到三角形'''
    iPDF = []
    for i in range(len(t)):
        if i == 0:
            iPDF.append(t[i][3])
        else:
            iPDF.append(iPDF[i - 1] + t[i][3])
    pts = []

    for _ in range(num):
        index = np.random.uniform(0, 1, 1)
        for i in range(len(iPDF)):
            if index <= iPDF[i]:
                index = i
                break
        pt = uniform_sample_in_triangle(t[index][0:3], 1)[0]
        pts.append(pt)

    return pts
