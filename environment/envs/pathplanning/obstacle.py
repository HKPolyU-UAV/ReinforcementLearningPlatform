import math
import random

import numpy as np


def sind(theta):
    return math.sin(theta / 180.0 * math.pi)


def cosd(theta):
    return math.cos(theta / 180.0 * math.pi)


class obstacle:
    def __init__(self, obs):
        self.name_set = ['triangle',  # 三角形(等腰)(就是因为描述方便)(格式统一)
                         'rectangle',  # 四边形(利用外接圆的方式去定义)
                         'pentagon',  # 五边形(正)(利用外接圆的方式去定义)
                         'hexagon',  # 六边形(正)(利用外接圆的方式去定义)
                         'heptagon'  # 七边形(正)(利用外接圆的方式去定义)
                         'octagon',  # 八边形(正)(利用外接圆的方式去定义)
                         'circle',  # 圆形
                         'ellipse']  # 椭圆形
        self.obs = self.set_obs(obs)  # the formation is ['name', [r], [points]]
        ''' triangle        ['triangle',  [pt1, pt2], [r, theta0, theta_bias]]     pt should be clockwise or counter-clock wise 
            rectangle       ['rectangle', [pt1, pt2], [r, theta0, theta_bias]]             pt1 and pt2 are the coordinate of the center
            pentagon        ['pentagon',  [pt1, pt2], [r, theta_bias]]
            hexagon         ['hexagon',   [pt1, pt2], [r, theta_bias]]
            heptagon        ['heptagon',  [pt1, pt2], [r, theta_bias]]
            octagon         ['octagon',   [pt1, pt2], [r, theta_bias]]
            circle          ['circle',    [pt1, pt2], [r]]
            ellipse         ['ellipse',   [pt1, pt2], [long_axis, short_axis, theta_bias]]'''

    @staticmethod
    def set_obs(message: list):
        obs = []
        if message is None:
            return obs
        if len(message) == 0:
            return obs
        for item in message:
            if not item:
                continue
            [name, [x, y], constraints] = item
            if name == 'triangle':  # ['triangle',  [pt1, pt2], [r, theta0, theta_bias]]
                [r, theta0, theta_bias] = constraints
                pt1 = [x + r * cosd(90 + theta_bias), y + r * sind(90 + theta_bias)]
                pt2 = [x + r * cosd(270 - theta0 + theta_bias), y + r * sind(270 - theta0 + theta_bias)]
                pt3 = [x + r * cosd(theta0 - 90 + theta_bias), y + r * sind(theta0 - 90 + theta_bias)]
                obs.append([name, list(np.around([x, y, r], 3)), list(np.around([pt1, pt2, pt3], 3))])
            elif name == 'rectangle':
                [r, theta0, theta_bias] = constraints
                pt1 = [x + r * cosd(theta0 + theta_bias), y + r * sind(theta0 + theta_bias)]
                pt2 = [x + r * cosd(180 - theta0 + theta_bias), y + r * sind(180 - theta0 + theta_bias)]
                pt3 = [x + r * cosd(180 + theta0 + theta_bias), y + r * sind(180 + theta0 + theta_bias)]
                pt4 = [x + r * cosd(-theta0 + theta_bias), y + r * sind(-theta0 + theta_bias)]
                obs.append([name, list(np.around([x, y, r], 3)), list(np.around([pt1, pt2, pt3, pt4], 3))])
            elif name == 'pentagon':
                [r, theta_bias] = constraints
                pt = []
                for i in range(5):
                    pt.append([x + r * cosd(90 + 72 * i + theta_bias), y + r * sind(90 + 72 * i + theta_bias)])
                obs.append([name, list(np.around([x, y, r], 3)), list(np.around(pt, 3))])
            elif name == 'hexagon':
                [r, theta_bias] = constraints
                pt = []
                for i in range(6):
                    pt.append([x + r * cosd(90 + 60 * i + theta_bias), y + r * sind(90 + 60 * i + theta_bias)])
                obs.append([name, list(np.around([x, y, r], 3)), list(np.around(pt, 3))])
            elif name == 'heptagon':
                [r, theta_bias] = constraints
                pt = []
                for i in range(7):
                    pt.append([x + r * cosd(90 + 360 / 7 * i + theta_bias), y + r * sind(90 + 360 / 7 * i + theta_bias)])
                obs.append([name, list(np.around([x, y, r], 3)), list(np.around(pt, 3))])
            elif name == 'octagon':
                [r, theta_bias] = constraints
                pt = []
                for i in range(8):
                    pt.append([x + r * cosd(90 + 45 * i + theta_bias), y + r * sind(90 + 45 * i + theta_bias)])
                obs.append([name, list(np.around([x, y, r], 3)), list(np.around(pt, 3))])
            elif name == 'circle':
                obs.append([name, list(np.around(constraints, 3)), list(np.around([x, y], 3))])
            elif name == 'ellipse':
                obs.append([name, list(np.around(constraints, 3)), list(np.around([x, y], 3))])
            else:
                print('Unknown obstacle type')
        return obs

    def get_obs(self):
        return self.obs

    @staticmethod
    def set_random_circle(xRange, yRange, rRange=None):
        if rRange is None:
            rRange = [0.5, 0.8]
        x = random.uniform(xRange[0], xRange[1])
        y = random.uniform(yRange[0], yRange[1])
        r = random.uniform(rRange[0], rRange[1])
        return ['circle', [x, y], [r]]

    @staticmethod
    def set_random_ellipse(xRange, yRange, longRange=None, shortRange=None, thetaMax=60):  # 都用的角度，这里也用角度把
        if longRange is None:
            longRange = [0.2, 0.4]
        if shortRange is None:
            shortRange = [0.2, 0.4]
        x = random.uniform(xRange[0], xRange[1])
        y = random.uniform(yRange[0], yRange[1])
        long = random.uniform(longRange[0], longRange[1])
        short = random.uniform(shortRange[0], shortRange[1])
        theta_bias = random.uniform(-thetaMax, thetaMax)
        return ['ellipse', [x, y], [long, short, theta_bias]]

    @staticmethod
    def set_random_poly(xRange, yRange, rRange=None, thetaMin=45, thetaMax=90, theta0Range=None):
        if theta0Range is None:
            theta0Range = [30, 60]
        if rRange is None:
            rRange = [0.5, 0.8]
        namelist = ['triangle', 'rectangle', 'pentagon', 'hexagon', 'heptagon', 'octagon']
        edge = random.sample([0, 1, 2, 3, 4, 5], 1)[0]
        x = random.uniform(xRange[0], xRange[1])
        y = random.uniform(yRange[0], yRange[1])
        r = random.uniform(rRange[0], rRange[1])
        theta_bias = random.uniform(thetaMin, thetaMax)
        theta0 = random.uniform(theta0Range[0], theta0Range[1])
        if edge == 0 or edge == 1:
            return [namelist[edge], [x, y], [r, theta0, theta_bias]]
        else:
            return [namelist[edge], [x, y], [r, theta_bias]]
