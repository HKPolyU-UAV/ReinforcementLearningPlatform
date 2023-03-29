import random
import cv2 as cv
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from environment.Color import Color
from environment.envs.pathplanning.obstacle import obstacle
from common.common_func import *


class samplingmap(obstacle):
    def __init__(self,
                 width: int = 400,
                 height: int = 400,
                 x_size: float = 10.,
                 y_size: float = 10.,
                 image_name: str = 'samplingmap',
                 start: list = None,
                 terminal: list = None,
                 obs=None,
                 draw=True):
        super(samplingmap, self).__init__(obs)  # 用obstacle的函数初始化sampling map
        self.width = width
        self.height = height
        self.x_size = x_size
        self.y_size = y_size
        self.name4image = image_name
        # self.start = [0.5, 0.5] if start is None else start
        # self.terminal = [x_size - 0.5, y_size - 0.5] if terminal is None else terminal
        self.start = start
        self.terminal = terminal
        self.obs = self.get_obs()
        self.obs_num = 0 if obs is None else len(obs)
        self.image = np.zeros([self.width, self.height, 3], np.uint8)
        self.image[:, :, 0] = np.ones([self.width, self.height]) * 255
        self.image[:, :, 1] = np.ones([self.width, self.height]) * 255
        self.image[:, :, 2] = np.ones([self.width, self.height]) * 255
        self.image_white = self.image.copy()  # 纯白图

        self.name4image = image_name
        # self.x_offset = self.width / 20  # leave blank for image
        # self.y_offset = self.height / 20

        self.x_offset = 2
        self.y_offset = 2

        self.pixel_per_meter = min((self.width - 2 * self.x_offset) / self.x_size,
                                   (self.height - 2 * self.y_offset) / self.y_size)

        self.map_draw_boundary()
        self.image_temp = self.image.copy()
        self.save = self.image.copy()
        # self.set_random_obstacles(10)
        self.map_draw(draw)

    def set_start(self, start):
        self.start = list(np.around(start, 3))

    def set_terminal(self, terminal):
        self.terminal = list(np.around(terminal, 3))

    def start_clip(self, _min: list, _max: list):
        self.start[0] = min(max(self.start[0], _min[0]), _max[0])
        self.start[1] = min(max(self.start[1], _min[1]), _max[1])

    def terminal_clip(self, _min: list, _max: list):
        self.terminal[0] = min(max(self.terminal[0], _min[0]), _max[0])
        self.terminal[1] = min(max(self.terminal[1], _min[1]), _max[1])

    def point_is_out(self, point: list) -> bool:
        """
        :brief:         if the robot is out
        :param point:   the position of the robot
        :return:        bool
        """
        return min(point) < -1e2 or point[0] > self.x_size + 1e-2 or point[1] > self.y_size + 1e-2

    def point_saturation(self, point):
        return [max(min(point[0], self.y_size - 1e-3), 1e-3), max(min(point[1], self.y_size - 1e-3), 1e-3)]

    def line_is_in_obs(self, point1: list, point2: list) -> bool:
        """
        :brief:             if a line segment has intersections with obstacles
        :param point1:      the first point of the line segment
        :param point2:      the second point of the line segment
        :return:            if the line segment has intersections with obstacles
        """
        for _obs in self.obs:
            if _obs[0] == 'circle':
                if line_is_in_circle(_obs[2], _obs[1][0], point1, point2):
                    return True
            elif _obs[0] == 'ellipse':
                if line_is_in_ellipse(_obs[1][0], _obs[1][1], _obs[1][2], _obs[2], point1, point2):
                    return True
            else:
                if line_is_in_poly([_obs[1][i] for i in [0, 1]], _obs[1][2], _obs[2], point1, point2):
                    return True
        return False

    def point_is_in_obs(self, point: list) -> bool:
        """
        :brief:             if a point is in obstacles
        :param point:       point
        :return:            if the point is in obstacles
        """
        for _obs in self.obs:
            if _obs[0] == 'circle':
                if point_is_in_circle(_obs[2], _obs[1][0], point):
                    return True
                else:
                    continue
            elif _obs[0] == 'ellipse':
                if point_is_in_ellipse(_obs[1][0], _obs[1][1], _obs[1][2], _obs[2], point):
                    return True
                else:
                    continue
            else:
                if point_is_in_poly([_obs[1][i] for i in [0, 1]], _obs[1][2], _obs[2], point):
                    return True
                else:
                    continue
        return False

    def pixel2dis(self, coord) -> list:
        """
        :brief:             the transformation between pixels in image and distance in physical world
        :param coord:       position in image coordinate
        :return:            position in physical world
        """
        x = (coord[0] - self.x_offset) / self.pixel_per_meter
        y = (self.height - self.y_offset - coord[1]) / self.pixel_per_meter
        return [x, y]

    def dis2pixel(self, coord) -> tuple:
        """
        :brief:         the transformation of coordinate between physical world and image
        :param coord:   position in physical world
        :return:        position in image coordinate
        """
        x = self.x_offset + coord[0] * self.pixel_per_meter
        y = self.height - self.y_offset - coord[1] * self.pixel_per_meter
        return int(x), int(y)

    def length2pixel(self, _l):
        """
        :brief:         the transformation of distance between physical world and image
        :param _l:      length in physical world
        :return:        length in image
        """
        return int(_l * self.pixel_per_meter)

    def test_func_point_is_in_obs_using_opencv_callback(self):
        """
        :brief:         as shown in the name of the function
        :return:        None
        """

        def callback(event, x, y, flags, param):
            self.image_temp = self.image.copy()
            if event == cv.EVENT_MOUSEMOVE:  # 鼠标左键抬起
                point = self.pixel2dis((x, y))
                cv.circle(self.image_temp, (x, y), 3, Color().DarkMagenta, -1)
                if min(point) <= 0. or point[0] > self.x_size or point[1] > self.y_size:
                    cv.putText(self.image_temp, "OUT", (x + 5, y + 5), cv.FONT_HERSHEY_SIMPLEX,
                               0.7, Color().DarkMagenta, 1, cv.LINE_AA)
                else:
                    cv.putText(self.image_temp, str(self.point_is_in_obs(point)), (x + 5, y + 5),
                               cv.FONT_HERSHEY_SIMPLEX,
                               0.7, Color().DarkMagenta, 1, cv.LINE_AA)

        cv.setMouseCallback(self.name4image, callback)
        while True:
            cv.imshow(self.name4image, self.image_temp)
            if cv.waitKey(1) == ord('q'):
                break
        cv.destroyAllWindows()

    def map_draw_boundary(self):
        cv.rectangle(self.image, self.dis2pixel([0., 0.]), self.dis2pixel([self.x_size, self.y_size]), Color().Black, 2)

    def map_draw_start_terminal(self):
        if (self.start is not None) and (self.terminal is not None):
            cv.circle(self.image, self.dis2pixel(self.start), self.length2pixel(0.15), Color().Red, -1)
            cv.circle(self.image, self.dis2pixel(self.terminal), self.length2pixel(0.15), Color().Blue, -1)
        else:
            print('No start point or terminal point')

    def map_draw_obs(self):
        if self.obs is None:
            print('No obstacles!!')
            return
        for [name, constraints, pts] in self.obs:  # [name, [], [pt1, pt2, pt3]]
            color = Color().DarkGray
            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if name == 'circle':
                cv.circle(self.image, self.dis2pixel(pts), self.length2pixel(constraints[0]), color, -1)
            elif name == 'ellipse':
                cv.ellipse(img=self.image,
                           center=self.dis2pixel(pts),
                           axes=(self.length2pixel(constraints[0]), self.length2pixel(constraints[1])),
                           angle=-constraints[2],
                           startAngle=0.,
                           endAngle=360.,
                           color=color,
                           thickness=-1)
            else:
                cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in pts]),
                                  color=color)

    def map_draw_photo_frame(self):
        cv.rectangle(self.image, (0, 0), (self.width - 1, self.dis2pixel([self.x_size, self.y_size])[1]), Color().White, -1)
        cv.rectangle(self.image, (0, 0), (self.dis2pixel([0., 0.])[0], self.height - 1), Color().White, -1)
        cv.rectangle(self.image, self.dis2pixel([self.x_size, self.y_size]), (self.width - 1, self.height - 1), Color().White, -1)
        cv.rectangle(self.image, self.dis2pixel([0., 0.]), (self.width - 1, self.height - 1), Color().White, -1)

    def map_draw(self, show=True, isWait=True):
        self.image = self.image_temp.copy()
        self.map_draw_obs()
        self.map_draw_photo_frame()
        self.map_draw_boundary()
        # self.map_draw_start_terminal()
        if show:
            cv.imshow(self.name4image, self.image)
            cv.waitKey(0) if isWait else cv.waitKey(1)

    def path_draw(self, path, name, color):
        """
        :param path:
        :param name:
        :param color:
        :return:
        """
        pt1 = path.pop()
        pt1_int = self.dis2pixel(pt1)
        while path:
            pt2 = path.pop()
            pt2_int = self.dis2pixel(pt2)
            cv.line(self.image, pt1_int, pt2_int, color, 2)
            # cv.imshow(self.name4image, self.image)
            # cv.waitKey(0)
            pt1 = pt2
            pt1_int = self.dis2pixel(pt1)
        self.map_draw_obs()
        self.map_draw_photo_frame()
        self.map_draw_boundary()
        self.map_draw_start_terminal()
        cv.imshow(self.name4image, self.image)
        cv.imwrite('../../../somefigures/figure/' + name, self.image)
        self.image = self.image_temp.copy()
        cv.waitKey(10)
        cv.destroyAllWindows()

    '''random obstacles'''

    def set_random_obs_single(self):
        """
        :return:
        """
        index = random.sample([0, 2, 3, 4, 5], 1)[0]  # 0-circle, 1-ellipse, 2-poly，大于1的数字越多，多边形的概率越大
        # index = 2
        if index == 0:
            newObs = self.set_random_circle(xRange=[0, self.x_size], yRange=[0, self.y_size], rRange=None)
            center = newObs[1]
            r = newObs[2][0]
        elif index == 1:
            newObs = self.set_random_ellipse(xRange=[0, self.x_size], yRange=[0, self.y_size], shortRange=None, longRange=None)
            center = newObs[1]
            r = max(newObs[2][0], newObs[2][1])
        else:
            newObs = self.set_random_poly(xRange=[0, self.x_size], yRange=[0, self.y_size], rRange=None, theta0Range=None)
            center = newObs[1]
            r = newObs[2][0]
        return newObs, center, r

    def set_random_obstacles(self, num):
        new_obs = []
        safety_dis = 0.4
        safety_dis_ST = 0.2
        for i in range(num):
            '''for each obstacle'''
            counter = 0
            while True:
                newObs, center, r = self.set_random_obs_single()  # 0-circle, 1-ellipse, 2-poly
                counter += 1
                if counter > 10000:
                    break
                is_acceptable = True
                '''检测newObs与起点和终点的距离'''
                if (self.start is not None) and (self.start != []) and (self.terminal is not None) and (self.terminal != []):
                    if (dis_two_points(self.start, center) < r + safety_dis_ST) or \
                            (dis_two_points(self.terminal, center) < r + safety_dis_ST):
                        continue
                '''检测newObs与起点和终点的距离'''

                '''检测障碍物与其他障碍物的距离'''
                for _obs in self.obs:
                    if _obs[0] == 'circle':
                        if dis_two_points(center, _obs[2]) < r + _obs[1][0] + safety_dis:
                            is_acceptable = False
                            break
                    elif _obs[0] == 'ellipse':
                        if dis_two_points(center, _obs[2]) < r + max(_obs[1][0], _obs[1][1]) + safety_dis:
                            is_acceptable = False
                            break
                    else:
                        if dis_two_points(center, [_obs[1][0], _obs[1][1]]) < r + _obs[1][2] + safety_dis:
                            is_acceptable = False
                            break
                '''检测障碍物与其他障碍物的距离'''
                if is_acceptable:
                    new_obs.append(newObs.copy())
                    break
            self.obs = self.set_obs(new_obs)

    @staticmethod
    def transfer_str_2_obs_info(string: str):
        string = string.replace(' ', '').replace("'", '').replace('[', '').replace(']', '').split(',')
        # obs_info = []
        # name_dict = ['circle', 'ellipse', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'heptagon', 'octagon']
        # print(string)
        name = string[0]
        if name == 'circle':
            r, x, y = float(string[1]), float(string[2]), float(string[3])
            obs_info = [name, [r], [x, y]]
        elif name == 'ellipse':
            long, short, theta, x, y = float(string[1]), float(string[2]), float(string[3]), float(string[4]), float(string[5])
            obs_info = [name, [long, short, theta], [x, y]]
        elif name == 'triangle':
            x, y, r = float(string[1]), float(string[2]), float(string[3])
            pts = [[float(string[4 + i * 2]), float(string[5 + i * 2])] for i in range(3)]
            obs_info = [name, [x, y, r], pts]
        elif name == 'rectangle':
            x, y, r = float(string[1]), float(string[2]), float(string[3])
            pts = [[float(string[4 + i * 2]), float(string[5 + i * 2])] for i in range(4)]
            obs_info = [name, [x, y, r], pts]
        elif name == 'pentagon':
            x, y, r = float(string[1]), float(string[2]), float(string[3])
            pts = [[float(string[4 + i * 2]), float(string[5 + i * 2])] for i in range(5)]
            obs_info = [name, [x, y, r], pts]
        elif name == 'hexagon':
            x, y, r = float(string[1]), float(string[2]), float(string[3])
            pts = [[float(string[4 + i * 2]), float(string[5 + i * 2])] for i in range(6)]
            obs_info = [name, [x, y, r], pts]
        elif name == 'heptagon':
            x, y, r = float(string[1]), float(string[2]), float(string[3])
            pts = [[float(string[4 + i * 2]), float(string[5 + i * 2])] for i in range(7)]
            obs_info = [name, [x, y, r], pts]
        elif name == 'octagon':
            x, y, r = float(string[1]), float(string[2]), float(string[3])
            pts = [[float(string[4 + i * 2]), float(string[5 + i * 2])] for i in range(8)]
            obs_info = [name, [x, y, r], pts]
        else:
            assert False
        return obs_info

    def map_create_continuous_database(self, map_num: int, filePath: str, fileName: str):
        """
        map_num:    number of the maps
        filePath:
        fileName:
        """
        f = open(file=filePath + fileName, mode='w')
        '''First part is the basic message'''
        f.writelines('x_size:' + str(self.x_size) + '\n')
        f.writelines('y_size:' + str(self.y_size) + '\n')
        '''First part is the basic message'''
        f.writelines('BEGIN' + '\n')
        for i in range(map_num):
            if i % 100 == 0:
                print('num:', i)
            self.set_start([random.uniform(0.15, self.x_size - 0.15), random.uniform(0.15, self.x_size - 0.15)])
            self.set_terminal([random.uniform(0.15, self.x_size - 0.15), random.uniform(0.15, self.x_size - 0.15)])
            self.set_random_obstacles(20)
            self.map_draw(show=True, isWait=False)
            '''Second part is the start-terminal message'''
            f.writelines('num' + str(i) + '\n')
            f.writelines('start:' + str(list(self.start)) + '\n')
            f.writelines('terminal:' + str(list(self.terminal)) + '\n')
            '''Second part is the start-terminal message'''

            '''Third part is the continuous obstacles' message'''
            f.writelines('obs num:' + str(len(self.obs)) + '\n')
            for _obs in self.obs:
                f.writelines(str(_obs).replace('array', '').replace('(', '').replace(')', '') + '\n')
            '''Third part is the continuous obstacles' message'''
        f.writelines('END' + '\n')
        f.close()

    def map_load_continuous_database(self, databaseFile):
        BIG_DATA_BASE = []
        f = open(databaseFile, mode='r')
        ''''检测文件头'''
        assert self.x_size == float(f.readline().strip('\n')[7:])
        assert self.y_size == float(f.readline().strip('\n')[7:])
        assert f.readline().strip('\n') == 'BEGIN'
        ''''检测文件头'''

        line = f.readline().strip('\n')
        while line != 'END':
            DATA = []

            start = f.readline().strip('\n').replace('start:[', '').replace(']', '').replace(' ', '').split(',')
            DATA.append([float(kk) for kk in start])
            terminal = f.readline().strip('\n').replace('terminal:[', '').replace(']', '').replace(' ', '').split(',')
            DATA.append([float(kk) for kk in terminal])

            obsNum = int(f.readline().strip('\n').replace('obs num:', ''))  # obstacles
            DATA.append(obsNum)
            obs_info = []
            while obsNum > 0:
                obs_info.append(self.transfer_str_2_obs_info(f.readline().strip('\n')))  # each obstacle
                obsNum -= 1
            DATA.append(obs_info)
            BIG_DATA_BASE.append(DATA)
            line = f.readline().strip('\n')
            if line != 'END':
                if int(line[3:]) % 100 == 0:
                    print('...loading env ', int(line[3:]), '...')
        f.close()
        return BIG_DATA_BASE

    def autoSetWithDataBase(self, mapData):
        self.start = mapData[0]
        self.terminal = mapData[1]
        self.obs = mapData[3]

    @staticmethod
    def merge_database(databases):
        merge = []
        for database in databases:
            merge += database
        return merge

    def pre_fill_bound_with_rectangles(self):
        self.obs = []
        rx = math.sqrt(0.25 ** 2 + self.x_size ** 2 / 4)
        ry = math.sqrt(0.25 ** 2 + self.y_size ** 2 / 4)
        self.obs.append(['rectangle',
                         [self.x_size / 2, 0.25, rx],
                         [[0, 0], [self.x_size, 0],[self.x_size, 0.5], [0, 0.5]]])
        self.obs.append(['rectangle',
                         [self.x_size - 0.25, self.y_size / 2, ry],
                         [[self.x_size - 0.5, 0], [self.x_size, 0], [self.x_size, self.y_size], [self.x_size - 0.5, self.y_size]]])
        self.obs.append(['rectangle',
                         [self.x_size / 2, self.y_size - 0.25, rx],
                         [[self.x_size, self.y_size - 0.5], [self.x_size - 0.5, self.y_size], [0, self.y_size], [0, self.y_size - 0.5]]])
        self.obs.append(['rectangle',
                         [0.25, self.y_size / 2, ry],
                         [[0.5, self.y_size], [0, self.y_size], [0, 0], [0.5, 0]]])
