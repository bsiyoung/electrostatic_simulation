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
        self.p1 = [x1, y1]
        self.p2 = [x2, y2]
        self.cntr = [(x1 + x2) / 2, (y1 + y2) / 2]

        self.norm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.u_vec = [(x2 - x1) / self.norm, (y2 - y1) / self.norm]

    def get_electric_potential(self, x, y):
        r = [x - self.cntr[0], y - self.cntr[1]]
        local_x = self.u_vec[0] * r[0] + self.u_vec[1] * r[1]  # dot product
        local_y = -self.u_vec[1] * r[0] + self.u_vec[0] * r[1]

        a = self.norm / 2
        buf1 = np.sqrt((local_x - a) ** 2 + local_y ** 2) - local_x + a
        buf2 = np.sqrt((local_x + a) ** 2 + local_y ** 2) - local_x - a

        res = self.density / (4 * pi * eps) * np.log(buf1 / buf2)

        return res


