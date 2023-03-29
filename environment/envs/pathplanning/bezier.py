import numpy as np
import matplotlib.pyplot as plt


class Bezier:
    def __init__(self, points):
        self.points = points
        self.nt = 100
        self.t = np.linspace(0, 1, self.nt)
        self.curve = np.atleast_2d([[0, 0]] * self.nt)

    @staticmethod
    def TwoPoints(t, P1, P2):
        return (1 - t) * P1 + t * P2

    def Points(self, t, points):
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [self.TwoPoints(t, points[i1], points[i1 + 1])]
        return newpoints

    def Point(self, t, points):
        while len(points) > 1:
            points = self.Points(t, points)
        return points[0]

    def Curve(self):
        curve = np.array([[0.0] * len(self.points[0])])
        for t in self.t:
            curve = np.append(curve, [self.Point(t, self.points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve

    def Sample(self, n):
        _t = np.linspace(0, 1, n)
        curve = np.array([[0.0] * len(self.points[0])])
        for __t in _t:
            curve = np.append(curve, [self.Point(__t, self.points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve

    def Draw(self):
        plt.figure(1)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.grid(b=True, which='major', axis='both')

        plt.plot(self.curve[:, 0], self.curve[:, 1])
        plt.plot(self.points[:, 0], self.points[:, 1], 'ro:')
        plt.show()
