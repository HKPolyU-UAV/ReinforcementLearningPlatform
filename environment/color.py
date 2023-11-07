import random


class Color:
    def __init__(self):
        """
        :brief:     初始化
        """
        self.Black = (0, 0, 0)
        self.White = (255, 255, 255)

        self.Blue = (255, 0, 0)
        self.Green = (0, 255, 0)
        self.Red = (0, 0, 255)

        self.Yellow = (0, 255, 255)
        self.Cyan = (255, 255, 0)
        self.Magenta = (255, 0, 255)

        self.DarkSlateBlue = (139, 61, 72)
        self.LightPink = (193, 182, 255)
        self.Orange = (0, 165, 255)
        self.DarkMagenta = (139, 0, 139)
        self.Chocolate2 = (33, 118, 238)
        self.Thistle = (216, 191, 216)
        self.Purple = (240, 32, 160)
        self.DarkGray = (169, 169, 169)
        self.Gray = (128, 128, 128)
        self.DimGray = (105, 105, 105)
        self.DarkGreen = (0, 100, 0)
        self.LightGray = (199, 199, 199)

        self.n_color = 20

        self.color_container = [self.Black,             # 黑色
                                self.White,             # 白色
                                self.Blue,              # 蓝色
                                self.Green,             # 绿色
                                self.Red,               # 红色
                                self.Yellow,            # 黄色
                                self.Cyan,              # 青色
                                self.Magenta,           # 品红
                                self.DarkSlateBlue,     # 深石板蓝
                                self.LightPink,         # 浅粉
                                self.Orange,            # 橘黄
                                self.DarkMagenta,       # 深洋红色
                                self.Chocolate2,        # 巧克力
                                self.Thistle,           # 蓟
                                self.Purple,            # 紫色
                                self.DarkGray,          # 深灰
                                self.Gray,              # 灰色
                                self.DimGray,           # 丁雷
                                self.LightGray,         # 浅灰
                                self.DarkGreen          # 深绿
                                ]

    def get_color_by_item(self, _n: int):
        """
        :brief:         通过序号索引颜色
        :param _n:      序号
        :return:        返回的颜色
        """
        assert 0 <= _n < self.n_color   # 当 _n 不满足时触发断言
        return self.color_container[_n]

    def random_color(self):
        return self.color_container[random.randint(0, self.n_color - 1)]

    @staticmethod
    def random_color_by_BGR():
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
