import numpy as np


def rastrigin(coor, A=10):
    np_coor = np.array(coor)
    n = len(coor)
    return A * n + np.sum(np.square(np_coor) - A * np.cos(2 * np.pi * np_coor))


def ackley(coor):
    x, y = coor
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) \
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x))) + np.e + 20


def sphere(coor):
    np_coor = np.array(coor)
    return np.sum(np_coor ** 2)


def beale(coor):
    x, y = coor
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * (y ** 2)) ** 2 + (2.625 - x + x * (y ** 3)) ** 2


def booth(coor):
    x, y = coor
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def bukin(coor):
    x, y = coor
    return 100 * np.sqrt(np.abs(y - 0.01 * np.square(x))) + 0.01 * np.abs(x + 10)


def levi(coor):
    x, y = coor
    return np.square(np.sin(3 * np.pi * x)) + np.square(x - 1) * (1 + np.square(3 * np.pi * y)) + \
        np.square(y - 1) * (1 + np.square(np.sin(2 * np.pi * y)))


def himmelblau(coor):
    x, y = coor
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def easom(coor):
    x, y = coor
    return -np.cos(x) * np.cos(y) * np.exp(-(np.square(x - np.pi) + np.square(y - np.pi)))


def mccormick(coor):
    x, y = coor
    return np.sin(x + y) + np.square(x - y) - 1.5 * x + 2.5 * y + 1


def three_hump_camel(coor):
    x, y = coor
    return 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6) / 6 + x * y + y ** 2


def cross_in_tray(coor):
    x, y = coor
    return -1e-4 * np.power(np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x ** 2 + y ** 2) / np.pi))), 0.1)


def holder(coor):
    x, y = coor
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi)))


class FunctionFactory:
    def __init__(self):
        self.functions = {
            'rastrigin': ((-5.12, 5.12, -5.12, 5.12), rastrigin),
            'ackley': ((-5, 5, -5, 5), ackley),
            'sphere': ((-10, 10, -10, 10), sphere),
            'beale': ((-4.5, 4.5, -4.5, 4.5), beale),
            'booth': ((-10, 10, -10, 10), booth),
            'bukin': ((-15, -5, -3, 3), bukin),
            'levi': ((-10, 10, -10, 10), levi),
            'himmelblau': ((-5, 5, -5, 5), himmelblau),
            'easom': ((-10, 10, -10, 10), easom),
            'mccormick': ((-1.5, 4, -3, 4), mccormick),
            'three_hump_camel': ((-5, 5, -5, 5), three_hump_camel),
            'cross_in_tray': ((-10, 10, -10, 10), cross_in_tray),
            'holder': ((-10, 10, -10, 10), holder)
        }

    def get_function(self, f_name):
        return self.functions.get(f_name, ((None, None, None, None), None))
