from typing import List

import numpy as np

from charge import LineChargeDist


ref_point: List | None = None  # if None, reference point is at infinity
ref_potential = 0

pi = np.pi
eps = 8.8541878128e-12

job_queue = {
    'queue': [],
    'canvas': []
}
charges = [
    LineChargeDist(-0.01, 0, 0.01, 0, 0.01)
]


def get_electric_potential(x, y):
    res = -ref_potential
    for charge in charges:
        res += charge.get_electric_potential(x, y)

    return res
