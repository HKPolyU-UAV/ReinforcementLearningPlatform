from utils.functions import *


class Map:
    def __init__(self):
        self.name_set = ['triangle',  # 三角形(等腰)(就是因为描述方便)(格式统一)
                         'rectangle',  # 四边形(利用外接圆的方式去定义)
                         'pentagon',  # 五边形(正)(利用外接圆的方式去定义)
                         'hexagon',  # 六边形(正)(利用外接圆的方式去定义)
                         'heptagon'  # 七边形(正)(利用外接圆的方式去定义)
                         'octagon',  # 八边形(正)(利用外接圆的方式去定义)
                         'circle',  # 圆形
                         'ellipse']  # 椭圆形
        self.obs = []
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

    @staticmethod
    def set_random_circle(xRange, yRange, rRange=None):
        if rRange is None:
            rRange = [0.5, 0.8]
        x = np.random.uniform(xRange[0], xRange[1])
        y = np.random.uniform(yRange[0], yRange[1])
        r = np.random.uniform(rRange[0], rRange[1])
        return ['circle', [x, y], [r]]

    @staticmethod
    def set_random_ellipse(xRange, yRange, longRange=None, shortRange=None, thetaMax=60):  # 都用的角度，这里也用角度把
        if longRange is None:
            longRange = [0.2, 0.4]
        if shortRange is None:
            shortRange = [0.2, 0.4]
        x = np.random.uniform(xRange[0], xRange[1])
        y = np.random.uniform(yRange[0], yRange[1])
        long = np.random.uniform(longRange[0], longRange[1])
        short = np.random.uniform(shortRange[0], shortRange[1])
        theta_bias = np.random.uniform(-thetaMax, thetaMax)
        return ['ellipse', [x, y], [long, short, theta_bias]]

    @staticmethod
    def set_random_poly(xRange, yRange, rRange=None, thetaMin=45, thetaMax=90, theta0Range=None):
        if theta0Range is None:
            theta0Range = [30, 60]
        if rRange is None:
            rRange = [0.5, 0.8]
        namelist = ['triangle', 'rectangle', 'pentagon', 'hexagon', 'heptagon', 'octagon']
        edge = np.random.sample([0, 1, 2, 3, 4, 5], 1)[0]
        x = np.random.uniform(xRange[0], xRange[1])
        y = np.random.uniform(yRange[0], yRange[1])
        r = np.random.uniform(rRange[0], rRange[1])
        theta_bias = np.random.uniform(thetaMin, thetaMax)
        theta0 = np.random.uniform(theta0Range[0], theta0Range[1])
        if edge == 0 or edge == 1:
            return [namelist[edge], [x, y], [r, theta0, theta_bias]]
        else:
            return [namelist[edge], [x, y], [r, theta_bias]]

    @staticmethod
    def generate_random_ST(xMax, yMax, safety_dis_st):
        start = np.random.uniform([0.3, 0.3], [xMax - 0.3, yMax - 0.3])
        terminal = start.copy()
        __k = 0
        while np.linalg.norm(terminal - start) < safety_dis_st:
            terminal = np.random.uniform([0.3, 0.3], [xMax - 0.3, yMax - 0.3])
            __k += 1
            assert __k < 10000
        return start, terminal

    @staticmethod
    def generate_one_circular_obs(xMax, yMax, rMin, rMax):
        center = np.random.uniform([0., 0.], [xMax, yMax])
        r = np.random.uniform(rMin, rMax)
        return ['circle', center, [r]]

    @staticmethod
    def is_new_obs_in_obs(obs, new_obs, safety_dis_obs):
        name1, c1, r1 = obs
        name2, c2, r2 = new_obs
        if name1 == 'ellipse' or name2 == 'ellipse':
            return True
        else:
            return dis_two_points(c1, c2) <= r1[0] + r2[0] + safety_dis_obs

    def is_obs_legal(self, s, t, obs, safety_dis_obs, safety_dis_st):
        name, center, constraints = obs
        if name == 'circle':
            if dis_two_points(s, center) <= constraints[0] + safety_dis_st:
                return False
            if dis_two_points(t, center) <= constraints[0] + safety_dis_st:
                return False
            for _o in self.obs:
                if self.is_new_obs_in_obs(obs=_o, new_obs=obs, safety_dis_obs=safety_dis_obs):
                    return False
            return True
        elif name == 'ellipse':
            return False
        else:
            return False

    def generate_circle_obs_training_dataset(self,
                                             xMax: float = 5.0,
                                             yMax: float = 5.0,
                                             safety_dis_obs: float = 0.4,
                                             safety_dis_st: float = 0.2,
                                             rMin: float = 0.1,
                                             rMax: float = 0.6,
                                             obsNum: int = -1,
                                             batch: int = 1000,
                                             filename: str = 'dataset.txt'):
        """
        Args:
            filename:
            xMax:               地图 x 尺寸
            yMax:               地图 y 尺寸
            safety_dis_obs:     障碍物之间的最短距离
            safety_dis_st:      起始点之间、起始点与障碍物的最短距离
            rMin:               圆最小半径
            rMax:               圆最大半径
            obsNum:             障碍物数量
            batch:
        Tips: This file generates a dataset for obstacle avoidance training, specifically, a *.txt file.
              For simplicity, all obstacles are set to be circular.
        """
        f = open(filename, 'w')
        f.writelines('MapSize: [%.2f.%.2f]' % (xMax, yMax) + '\n')
        f.writelines('BEGIN' + '\n')
        for i in range(batch):
            f.writelines('=====%.0f=====' % i + '\n')
            start, terminal = self.generate_random_ST(xMax, yMax, safety_dis_st)
            f.writelines('Start: [%.2f, %.2f], Terminal: [%.2f, %.2f]' % (start[0], start[1], terminal[0], terminal[1]) + '\n')
            _obsNum = obsNum if 0 < obsNum < 20 else np.random.randint(1, 16)
            f.writelines('ObsNum: %.0f' % _obsNum + '\n')
            for _ in range(_obsNum):
                _new_obs = ['circle', start.copy(), [0.1]]
                while not self.is_obs_legal(start, terminal, _new_obs, safety_dis_obs, safety_dis_st):
                    _new_obs = self.generate_one_circular_obs(xMax, yMax, rMin, rMax)
                f.writelines('[' + _new_obs[0] + ', ' + '[%.2f], ' % (_new_obs[2][0]) + '[%.2f, %.2f]' % (_new_obs[1][0], _new_obs[1][1]) + ']\n')
                self.obs.append(_new_obs)
            f.writelines('=====%.0f=====' % i + '\n')
            f.writelines('\n')
        f.writelines('END' + '\n')
        f.close()
