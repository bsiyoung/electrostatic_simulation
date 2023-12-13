import numpy as np


pi = np.pi
eps = 8e-12


class LineChargeDist:
    def __init__(self, x1, y1, x2, y2, density):
        self.p1 = None
        self.p2 = None
        self.cntr = None
        self.u_vec = None
        self.norm = 0
        self.set_points(x1, y1, x2, y2)

        self.density = density

    def set_points(self, x1, y1, x2, y2):
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        self.cntr = (self.p1 + self.p2) / 2

        r_sub = self.p2 - self.p1
        self.norm = np.sqrt(np.dot(r_sub, r_sub))
        self.u_vec = r_sub / self.norm

    def get_electric_potential(self, x, y):
        target = np.array([x, y])

        v_local_r = target - self.cntr
        local_x = np.dot(self.u_vec, v_local_r)
        v_local_x = self.u_vec * local_x
        v_local_y = v_local_r - v_local_x
        local_y = np.sqrt(np.dot(v_local_y, v_local_y))

        return np.sqrt(local_x ** 2 + local_y ** 2)

        a = self.norm / 2
        buf1 = np.sqrt((local_x - a) ** 2 + local_y ** 2) - local_x + a
        buf2 = np.sqrt((local_x + a) ** 2 + local_y ** 2) - local_x - a

        if buf2 == 0:
            print()

        res = self.density / (4 * pi * eps) * np.log(buf1 / buf2)

        return res


