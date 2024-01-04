from typing import Tuple

import math

import numpy as np
from numba import cuda

pi = np.pi
eps = 8.8541878128e-12  # electric permittivity of vacuum

FORM_CONSTANT = 0


class ChargeDist:
    def __init__(self, x1, y1, x2, y2,
                 density=1e-8, depth=1e-1,
                 form='constant'):
        self.p1: np.ndarray | None = None
        self.p2: np.ndarray | None = None
        self.cntr: np.ndarray | None = None
        self.norm: float = 0.0
        self.u_vec: Tuple[np.ndarray, np.ndarray] | None = None

        self.set_points(x1, y1, x2, y2)
        self.density = density
        self.depth = depth
        self.form = form

    def set_points(self, x1, y1, x2, y2):
        self.p1 = np.array((x1, y1), dtype=np.float32)
        self.p2 = np.array((x2, y2), dtype=np.float32)
        self.cntr = (self.p1 + self.p2) / 2

        p1_to_p2 = self.p2 - self.p1
        self.norm = np.sqrt(np.dot(p1_to_p2, p1_to_p2))

        u_vec_0 = p1_to_p2 / self.norm
        self.u_vec = (  # Unit vectors
            u_vec_0,  # + local x direction
            np.array((-u_vec_0[1], u_vec_0[0]), dtype=np.float32)  # + local y direction
        )

    def get_potential(self, x, y):
        cntr_to_r = np.array((x, y), dtype=np.float32) - self.cntr

        # Local position
        local_x = np.dot(cntr_to_r, self.u_vec[0])
        local_y = np.dot(cntr_to_r, self.u_vec[1])

        if self.form == 'constant':
            return self.__calc_constant_0(local_x, local_y)

    def __calc_constant_0(self, x, y):
        half_depth = self.depth / 2

        buf = self.__calc_constant_1(x, y, half_depth) - self.__calc_constant_1(x, y, -half_depth)
        return self.density / (4 * pi * eps) * buf

    def __calc_constant_1(self, x, y, z):
        a = self.norm / 2

        buf00 = x ** 2 + y ** 2 + z ** 2
        buf01 = a - x
        buf02 = a + x

        buf10 = a ** 2 - 2 * a * x + buf00
        buf11 = a ** 2 + 2 * a * x + buf00

        buf20 = (buf01 * np.log(np.sqrt(buf10) + z)) if buf01 != 0 else 0
        buf21 = (buf02 * np.log(np.sqrt(buf11) + z)) if buf02 != 0 else 0
        buf22 = (y * np.arctan((z * buf01) / (y * np.sqrt(buf10)))) if y != 0 else 0
        buf23 = (y * np.arctan((z * buf02) / (y * np.sqrt(buf11)))) if y != 0 else 0
        buf24 = (z * np.arctanh(buf01 / np.sqrt(buf01 ** 2 + y ** 2 + z ** 2))) if z != 0 else 0
        buf25 = (z * np.arctanh(buf02 / np.sqrt(buf02 ** 2 + y ** 2 + z ** 2))) if z != 0 else 0

        return buf20 + buf21 - buf22 - buf23 + buf24 + buf25


@cuda.jit
def gpu_get_potential(r_x, r_y, charge):
    cntr_x = (charge[0] + charge[2]) / 2
    cntr_y = (charge[1] + charge[3]) / 2
    cntr_to_r_x = r_x - cntr_x
    cntr_to_r_y = r_y - cntr_y

    u1_x = charge[4]
    u1_y = charge[5]
    u2_x = charge[6]
    u2_y = charge[7]

    local_x = cntr_to_r_x * u1_x + cntr_to_r_y * u1_y
    local_y = cntr_to_r_x * u2_x + cntr_to_r_y * u2_y

    if charge[11] == FORM_CONSTANT:
        return gpu_calc_constant_0(local_x, local_y, charge)

    return 0


@cuda.jit
def gpu_calc_constant_0(x, y, charge):
    half_depth = charge[10] / 2
    dx = charge[2] - charge[0]
    dy = charge[3] - charge[1]
    norm = math.sqrt(dx ** 2 + dy ** 2)

    buf0 = gpu_calc_constant_1(x, y, half_depth, norm)
    buf1 = gpu_calc_constant_1(x, y, -half_depth, norm)
    buf = buf0 - buf1

    return charge[8] / (4 * pi * eps) * buf


@cuda.jit
def gpu_calc_constant_1(x, y, z, norm):
    a = norm / 2

    buf00 = x ** 2 + y ** 2 + z ** 2
    buf01 = a - x
    buf02 = a + x

    buf10 = a ** 2 - 2 * a * x + buf00
    buf11 = a ** 2 + 2 * a * x + buf00

    buf20 = (buf01 * math.log(math.sqrt(buf10) + z)) if buf01 != 0 else 0
    buf21 = (buf02 * math.log(math.sqrt(buf11) + z)) if buf02 != 0 else 0
    buf22 = (y * math.atan((z * buf01) / (y * math.sqrt(buf10)))) if y != 0 else 0
    buf23 = (y * math.atan((z * buf02) / (y * math.sqrt(buf11)))) if y != 0 else 0
    buf24 = (z * math.atanh(buf01 / math.sqrt(buf01 ** 2 + y ** 2 + z ** 2))) if z != 0 else 0
    buf25 = (z * math.atanh(buf02 / math.sqrt(buf02 ** 2 + y ** 2 + z ** 2))) if z != 0 else 0

    return buf20 + buf21 - buf22 - buf23 + buf24 + buf25
