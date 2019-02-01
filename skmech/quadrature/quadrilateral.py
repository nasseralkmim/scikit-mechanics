"""quadrilateral weights and points for quadrature
"""
import numpy as np


class Quadrilateral(object):
    """Define weights and points for gaussian quadrature in quadrilateral
    elements

    Args:
        num_of_points (int): quadrature num_of_points

    Attrubutes:
        weights (list): list of weights
        points (list): points to evaluate functions

    """

    def __init__(self, num_of_points):
        if num_of_points == 2:
            self.weights = 4 * [1.0]
            p = np.sqrt(1 / 3)
            self.points = [[-p, -p], [p, -p], [p, p], [-p, p]]

        elif num_of_points == 3:
            self.weights = []
            self.points = []
            p = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
            w = [5 / 9, 8 / 9, 5 / 9]
            for wi, pi in zip(w, p):
                for wj, pj in zip(w, p):
                    self.weights.append(wi * wj)
                    self.points.append([pi, pj])

        elif num_of_points == 4:
            self.weights = []
            self.points = []
            p1 = np.sqrt(3 / 7 - 2 / 7 * (np.sqrt(6 / 5)))
            p2 = np.sqrt(3 / 7 + 2 / 7 * (np.sqrt(6 / 5)))
            w1 = (18 + np.sqrt(30.)) / 36
            w2 = (18 - np.sqrt(30.)) / 36
            p = [-p1, p1, -p2, p2]
            w = [w1, w1, w2, w2]
            for wi, pi in zip(w, p):
                for wj, pj in zip(w, p):
                    self.weights.append(wi * wj)
                    self.points.append([pi, pj])

        elif num_of_points == 8:
            self.weights = []
            self.points = []
            w = [
                0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                0.3137066458778873, 0.2223810344533745, 0.2223810344533745,
                0.1012285362903763, 0.1012285362903763
            ]
            p = [
                -0.1834346424956498, 0.1834346424956498, -0.5255324099163290,
                0.5255324099163290, -0.7966664774136267, 0.7966664774136267,
                -0.9602898564975363, 0.9602898564975363
            ]
            for wi, pi in zip(w, p):
                for wj, pj in zip(w, p):
                    self.weights.append(wi * wj)
                    self.points.append([pi, pj])

        self.num = len(self.points)


if __name__ is '__main__':
    quad = Quadrilateral(3)
    print(quad.points)
    print(quad.weights)
