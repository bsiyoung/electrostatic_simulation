import numpy as np


pi = np.pi
eps = 8.8541878128e-12  # electric permittivity of vacuum


class LineChargeDist:
    def __init__(self, x1, y1, x2, y2, density, depth=1e-1):
        self.p1 = None
        self.p2 = None
        self.cntr = None
        self.u_vec = None
        self.norm = 0
        self.set_points(x1, y1, x2, y2)

        self.density = density
        self.depth = depth

    def set_points(self, x1, y1, x2, y2):
        self.p1 = np.array([x1, y1], dtype=np.float64)
        self.p2 = np.array([x2, y2], dtype=np.float64)
        self.cntr: np.ndarray = (self.p1 + self.p2) / 2

        r_sub: np.ndarray = self.p2 - self.p1
        self.norm: np.float64 = np.sqrt(np.dot(r_sub, r_sub))
        self.u_vec: np.ndarray = r_sub / self.norm

    def get_electric_potential(self, x, y):
        target = np.array([x, y])

        v_local_r = target - self.cntr
        local_x = np.dot(self.u_vec, v_local_r)
        v_local_x = self.u_vec * local_x
        v_local_y = v_local_r - v_local_x
        local_y = np.sqrt(np.dot(v_local_y, v_local_y))

        res = self.density / (4 * pi * eps) * self.__calc_0(local_x, local_y, self.norm / 2, self.depth / 2)

        return res

    @staticmethod
    def __calc_0(x, y, a, b):
        return LineChargeDist.__calc_1(x, y, a, b) - LineChargeDist.__calc_1(x, y, a, -b)

    @staticmethod
    def __calc_1(x, y, a, z):
        buf00 = x ** 2 + y ** 2 + z ** 2
        buf01 = a - x
        buf02 = a + x

        buf10 = a ** 2 - 2 * a * x + buf00
        buf11 = a ** 2 + 2 * a * x + buf00

        buf20 = (buf01 * np.log(np.sqrt(buf10) + z)) if buf01 > 1e-16 else 0
        buf21 = (buf02 * np.log(np.sqrt(buf11) + z)) if buf02 > 1e-16 else 0
        buf22 = (y * np.arctan((z * buf01)/(y * np.sqrt(buf10)))) if y != 0 else 0
        buf23 = (y * np.arctan((z * buf02)/(y * np.sqrt(buf11)))) if y != 0 else 0
        buf24 = (z * np.arctanh(buf01 / np.sqrt(buf01 ** 2 + y ** 2 + z ** 2))) if z != 0 else 0
        buf25 = (z * np.arctanh(buf02 / np.sqrt(buf02 ** 2 + y ** 2 + z ** 2))) if z != 0 else 0

        return buf20 + buf21 - buf22 - buf23 + buf24 + buf25
