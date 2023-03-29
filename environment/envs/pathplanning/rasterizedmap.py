import os
import sys

import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../")
from environment.Color import Color
from environment.envs.pathplanning.samplingmap import samplingmap
from common.common_func import *


class rasterizedmap(samplingmap):
    def __init__(self, width, height, x_size, y_size, image_name, start, terminal, obs, draw, x_grid, y_grid):
        super(rasterizedmap, self).__init__(width, height, x_size, y_size, image_name, start, terminal, obs, draw)
        self.x_grid = x_grid  # x栅格数
        self.y_grid = y_grid  # y栅格数
        self.x_meter_per_grid = self.x_size / self.x_grid  # x每格对应的实际距离(米)
        self.y_meter_per_grid = self.y_size / self.y_grid  # y每格对应的实际距离(米)
        self.x_pixel_per_grid = self.pixel_per_meter * self.x_meter_per_grid  # x每格对应的实际长度(像素)
        self.y_pixel_per_grid = self.pixel_per_meter * self.y_meter_per_grid  # y每格对应的实际长度(像素)
        self.map_flag = [[0 for _ in range(x_grid)] for _ in range(y_grid)]

        self.map_rasterization()
        self.draw_rasterization_map(isShow=False)

    def is_grid_has_obs(self, points: list) -> int:
        for _point in points:
            if self.point_is_in_obs(_point):
                return 1
        '''四个顶点都不在障碍物里面'''

        assert len(points) == 4
        for i in range(4):
            if self.line_is_in_obs(points[i % 4], points[(i + 1) % 4]):
                return 1
        '''四个边都不在障碍物里面'''

        for _obs in self.obs:
            c1 = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]
            if self.point_is_in_obs(c1):
                return 1
        '''格子不在障碍物里面'''
        return 0

    @staticmethod
    def is_grid_has_single_obs(points, obs):
        if obs[0] == 'circle':
            for _point in points:
                if point_is_in_circle(obs[2], obs[1][0], _point):
                    return 1
            assert len(points) == 4
            for i in range(4):
                if line_is_in_circle(obs[2], obs[1][0], points[i % 4], points[(i + 1) % 4]):
                    return 1
            c1 = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]
            if point_is_in_circle(obs[2], obs[1][0], c1):
                return 1
            return 0
        elif obs[0] == 'ellipse':
            for _point in points:
                if point_is_in_ellipse(obs[1][0], obs[1][1], obs[1][2], obs[2], _point):
                    return 1
            assert len(points) == 4
            for i in range(4):
                if line_is_in_ellipse(obs[1][0], obs[1][1], obs[1][2], obs[2], points[i % 4], points[(i + 1) % 4]):
                    return 1
            c1 = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]
            if point_is_in_ellipse(obs[1][0], obs[1][1], obs[1][2], obs[2], c1):
                return 1
            return 0
        else:
            for _point in points:
                if point_is_in_poly([obs[1][0], obs[1][1]], obs[1][2], obs[2], _point):
                    return 1
            assert len(points) == 4
            for i in range(4):
                if line_is_in_poly([obs[1][0], obs[1][1]], obs[1][2], obs[2], points[i % 4], points[(i + 1) % 4]):
                    return 1
            c1 = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]
            if point_is_in_poly([obs[1][0], obs[1][1]], obs[1][2], obs[2], c1):
                return 1
            return 0

    @staticmethod
    def is_grid_has_single_obs2(points, obs):
        c1 = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]
        if obs[0] == 'circle':
            if point_is_in_circle(obs[2], obs[1][0], c1):
                return 1
            return 0
        elif obs[0] == 'ellipse':
            if point_is_in_ellipse(obs[1][0], obs[1][1], obs[1][2], obs[2], c1):
                return 1
            return 0
        else:
            if point_is_in_poly([obs[1][0], obs[1][1]], obs[1][2], obs[2], c1):
                return 1
            return 0

    @staticmethod
    def is_grid_has_single_obs3(points, obs):
        count = 0
        if obs[0] == 'circle':
            for _point in points:
                if point_is_in_circle(obs[2], obs[1][0], _point):
                    count += 1
                    # return 1
            assert len(points) == 4
            for i in range(4):
                if line_is_in_circle(obs[2], obs[1][0], points[i % 4], points[(i + 1) % 4]):
                    count += 1
                    # return 1
            if point_is_in_circle(obs[2], obs[1][0], [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]):
                return 1
            if count >= 3:
                return 1
            return 0
        elif obs[0] == 'ellipse':
            for _point in points:
                if point_is_in_ellipse(obs[1][0], obs[1][1], obs[1][2], obs[2], _point):
                    count += 1
                    # return 1
            assert len(points) == 4
            for i in range(4):
                if line_is_in_ellipse(obs[1][0], obs[1][1], obs[1][2], obs[2], points[i % 4], points[(i + 1) % 4]):
                    count += 1
                    # return 1
            if point_is_in_ellipse(obs[1][0], obs[1][1], obs[1][2], obs[2], [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]):
                return 1
            if count >= 3:
                return 1
            return 0
        else:
            for _point in points:
                if point_is_in_poly([obs[1][0], obs[1][1]], obs[1][2], obs[2], _point):
                    count += 1
                    # return 1
            assert len(points) == 4
            for i in range(4):
                if line_is_in_poly([obs[1][0], obs[1][1]], obs[1][2], obs[2], points[i % 4], points[(i + 1) % 4]):
                    count += 1
                    # return 1
            if point_is_in_poly([obs[1][0], obs[1][1]], obs[1][2], obs[2], [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[3][1]) / 2]):
                return 1
            if count >= 3:
                return 1
            return 0

    def map_rasterization(self):
        self.map_flag = [[0 for _ in range(self.x_grid)] for _ in range(self.y_grid)]
        for _obs in self.obs:
            obsName = _obs[0]
            if obsName == 'circle' or obsName == 'ellipse':
                x, y, r = _obs[2][0], _obs[2][1], _obs[1][0]
            else:
                x, y, r = _obs[1][0], _obs[1][1], _obs[1][2]
            up = self.point_in_grid(self.point_saturation([x, y + r]))[1]  # up
            down = self.point_in_grid(self.point_saturation([x, y - r]))[1]  # down
            left = self.point_in_grid(self.point_saturation([x - r, y]))[0]  # left
            right = self.point_in_grid(self.point_saturation([x + r, y]))[0]  # right
            for i in np.arange(left, right + 1, 1):
                for j in np.arange(down, up + 1, 1):
                    rec = [[i * self.x_meter_per_grid, j * self.y_meter_per_grid],
                           [(i + 1) * self.x_meter_per_grid, j * self.y_meter_per_grid],
                           [(i + 1) * self.x_meter_per_grid, (j + 1) * self.y_meter_per_grid],
                           [i * self.x_meter_per_grid, (j + 1) * self.y_meter_per_grid]]
                    # self.map_flag[i][j] = self.is_grid_has_single_obs(rec, _obs)
                    # self.map_flag[i][j] = self.is_grid_has_single_obs2(rec, _obs)
                    self.map_flag[i][j] = self.is_grid_has_single_obs3(rec, _obs)

    def draw_rasterization_map(self, isShow=True, isWait=True):
        self.image = self.image_temp.copy()
        self.map_draw_gird_rectangle()
        self.map_draw_x_grid()
        self.map_draw_y_grid()
        self.map_draw_start_terminal()
        self.map_draw_obs()
        self.map_draw_photo_frame()
        self.map_draw_boundary()
        if isShow:
            cv.imshow(self.name4image, self.image)
            cv.waitKey(0) if isWait else cv.waitKey(1)

    def map_draw_gird_rectangle(self):
        for i in range(self.x_grid):
            for j in range(self.y_grid):
                if self.map_flag[i][j] == 1:
                    pt1 = self.grid2pixel(coord_int=[i, j], pos='left-bottom', xoffset=-0, yoffset=0)
                    pt2 = self.grid2pixel(coord_int=[i, j], pos='right-top', xoffset=0, yoffset=0)
                    cv.rectangle(self.image, pt1, pt2, Color().LightGray, -1)

    def grid2pixel(self, coord_int: list, pos: str, xoffset=0, yoffset=0) -> tuple:
        """
        :brief:             to transfer grid in map to pixel in image
        :param coord_int:   coordinate [int, int]
        :param pos:         left-top, left-bottom, right-top. right-bottom
        :param xoffset:     xoffset
        :param yoffset:     yoffset
        :return:            pixel [int, int] (left-bottom)
        :tips:              the direction of offset is the same as that of image rather than real world or grid map
        """
        x = self.x_offset + coord_int[0] * self.x_pixel_per_grid
        y = self.height - self.y_offset - coord_int[1] * self.y_pixel_per_grid  # sef default to left-bottom

        if pos == 'left-bottom':
            return int(x) + xoffset, int(y) + yoffset
        elif pos == 'left-top':
            return int(x) + xoffset, int(y - self.y_pixel_per_grid) + yoffset
        elif pos == 'right-bottom':
            return int(x + self.x_pixel_per_grid) + xoffset, int(y) + yoffset
        elif pos == 'right-top':
            return int(x + self.x_pixel_per_grid) + xoffset, int(y - self.y_pixel_per_grid) + yoffset
        else:
            print('FUNCTION <grid2pixel>--ERROR input')
            return ()

    def map_draw_x_grid(self):
        for i in range(self.y_grid + 1):
            pt1 = self.grid2pixel(coord_int=[0, i], pos='left-bottom')
            pt2 = self.grid2pixel(coord_int=[self.x_grid, i], pos='left-bottom')
            cv.line(self.image, pt1, pt2, Color().Black, 1)

    def map_draw_y_grid(self):
        for i in range(self.x_grid + 1):
            pt1 = self.grid2pixel(coord_int=[i, 0], pos='left-bottom')
            pt2 = self.grid2pixel(coord_int=[i, self.y_grid], pos='left-bottom')
            cv.line(self.image, pt1, pt2, Color().Black, 1)

    def point_in_grid(self, point: list) -> list:
        if self.point_is_out(point):
            return [-1, -1]
        return [min(max(int(point[0] / self.x_meter_per_grid), 0), self.x_grid - 1), min(max(int(point[1] / self.y_meter_per_grid), 0), self.y_grid - 1)]

    def grid_2_point(self, grid):
        x, y = grid[0], grid[1]
        return [[x * self.x_meter_per_grid, y * self.y_meter_per_grid],                 # left-bottom
                [(x + 1) * self.x_meter_per_grid, y * self.y_meter_per_grid],           # right-bottom
                [(x + 1) * self.x_meter_per_grid, (y + 1) * self.y_meter_per_grid],     # right-top
                [x * self.x_meter_per_grid, (y + 1) * self.y_meter_per_grid]            # left-top
                ]

    def grid_center_point(self, grid):
        return [(grid[0] + 0.5) * self.x_meter_per_grid, (grid[1] + 0.5) * self.y_meter_per_grid]

    def is_grid_available(self, grid: list) -> bool:
        return True if self.map_flag[grid[0]][grid[1]] == 0 else False

    def map_create_database(self, map_num: int, filePath: str, fileName: str):
        """
        map_num:    number of the maps
        filePath:
        fileName:
        """
        f = open(file=filePath + fileName, mode='w')
        '''First part is the basic message'''
        f.writelines('x_size:' + str(self.x_size) + '\n')
        f.writelines('y_size:' + str(self.y_size) + '\n')
        f.writelines('x_grid:' + str(self.x_grid) + '\n')
        f.writelines('y_grid:' + str(self.y_grid) + '\n')
        '''First part is the basic message'''
        f.writelines('BEGIN' + '\n')
        for i in range(map_num):
            print('num:', i)
            self.set_start([random.uniform(0.15, self.x_size - 0.15), random.uniform(0.15, self.x_size - 0.15)])
            self.set_terminal([random.uniform(0.15, self.x_size - 0.15), random.uniform(0.15, self.x_size - 0.15)])
            self.set_random_obstacles(20)
            self.map_rasterization()
            self.draw_rasterization_map(isShow=True, isWait=False)
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

            '''Fourth part is the binary grid map'''
            f.writelines(str(self.map_flag).replace(', ', '').replace('[', '').replace(']', '') + '\n')
            '''Fourth part is the binary grid map'''
        f.writelines('END' + '\n')
        f.close()

    def map_load_database(self, databaseFile):
        BIG_DATA_BASE = []
        f = open(databaseFile, mode='r')
        ''''检测文件头'''
        assert self.x_size == float(f.readline().strip('\n')[7:])
        assert self.y_size == float(f.readline().strip('\n')[7:])
        assert self.x_grid == int(f.readline().strip('\n')[7:])
        assert self.y_grid == int(f.readline().strip('\n')[7:])
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
            flag = [[0 for _ in range(self.x_grid)] for _ in range(self.y_grid)]
            binary = f.readline().strip('\n')
            for i in range(self.x_grid * self.y_grid):
                col = i % self.y_grid  # 行数
                row = i // self.y_grid  # 列数
                flag[row][col] = int(binary[i])
            DATA.append(flag)
            BIG_DATA_BASE.append(DATA)
            line = f.readline().strip('\n')
            if line != 'END':
                if int(line[3:]) % 100 == 0:
                    print('...loading env ', int(line[3:]), '...')
        f.close()
        return BIG_DATA_BASE

    def test4database(self):
        DataBase = []
        names = os.listdir('10X10-40x40-DataBase')
        for name in names:
            print('Start Loading' + name)
            DataBase.append(self.map_load_database('10X10-40x40-DataBase/' + name))
            print('Finish Loading' + name)
        for database in DataBase:
            # print('new')
            for data in database:
                self.start = data[0]
                self.terminal = data[1]
                self.obs = data[3]
                self.map_flag = data[4]
                self.draw_rasterization_map(isShow=True, isWait=False)
