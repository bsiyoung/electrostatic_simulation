from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import List, Tuple

import numpy as np
from DrawManager import DrawRect
from charge import LineChargeDist

if TYPE_CHECKING:
    from DrawManager import DrawManager, DrawRect


class CalcJob:
    def __init__(self, scr_rect: Tuple[int], phy_rect: Tuple[float], ref_voltage: float):
        self.scr_rect = scr_rect
        self.phy_rect = phy_rect

        self.t_scr_x, self.t_phy_x = self.__get_target_pos(scr_rect[0], scr_rect[2], phy_rect[0], phy_rect[2])
        self.t_scr_y, self.t_phy_y = self.__get_target_pos(scr_rect[1], scr_rect[3], phy_rect[1], phy_rect[3])
        self.ref_voltage = ref_voltage

    @staticmethod
    def __get_target_pos(scr_st, scr_en, phy_st, phy_en):
        target_scr = (scr_st + scr_en) // 2
        target_phy = (phy_en + phy_st) / 2

        return target_scr, target_phy

    def do(self, charges: List[LineChargeDist]) -> Tuple[float, Tuple[int], Tuple]:
        res = -self.ref_voltage
        for charge in charges:
            res += charge.get_electric_potential(self.t_phy_x, self.t_phy_y)

        sub_rect = (
            (  # Left Top
                (self.scr_rect[0], self.scr_rect[1], self.t_scr_x, self.t_scr_y),
                (self.phy_rect[0], self.phy_rect[1], self.t_phy_x, self.t_phy_y)
            ),
            (  # Right Top
                (self.t_scr_x + 1, self.scr_rect[1], self.scr_rect[2], self.t_scr_y),
                (self.t_phy_x, self.phy_rect[1], self.phy_rect[2], self.t_phy_y)
            ),
            (  # Left Bottom
                (self.scr_rect[0], self.t_scr_y + 1, self.t_scr_x, self.scr_rect[3]),
                (self.phy_rect[0], self.t_phy_y, self.t_phy_x, self.phy_rect[3])
            ),
            (  # Right Bottom
                (self.t_scr_x + 1, self.t_scr_y + 1, self.scr_rect[2], self.scr_rect[3]),
                (self.t_phy_x, self.t_phy_y, self.phy_rect[2], self.phy_rect[3])
            )
        ) if (self.scr_rect[0] != self.scr_rect[2] or self.scr_rect[1] != self.scr_rect[3]) else None

        return res, self.scr_rect, sub_rect


class CalcManager:
    def __init__(self, draw_mgr: DrawManager, exit_flag: List[bool]):
        self.draw_mgr: DrawManager = draw_mgr
        self.exit_flag = exit_flag

        self.charges: List[LineChargeDist] = [LineChargeDist(-0.1, 0.05, 0.1, 0.05, 1e+8),
                                              LineChargeDist(-0.1, -0.05, 0.1, -0.05, -1e+8)]
        self.calc_queue: List[CalcJob] = []

    def add_charge(self, charge: LineChargeDist):
        self.charges.append(charge)

    def get_inf_ref_electric_potential(self, x, y):
        res = 0
        for charge in self.charges:
            res += charge.get_electric_potential(x, y)

        return res

    def loop(self):
        total_max_abs = -1
        while self.exit_flag[0] is False:
            time.sleep(0.01)

            idx = 0
            sub_max_abs = total_max_abs
            while idx < len(self.calc_queue) and idx < 10:
                calc_job = self.calc_queue[idx]
                res, rect, sub_rect = calc_job.do(self.charges)
                if sub_max_abs < abs(res):
                    sub_max_abs = abs(res)

                self.draw_mgr.add_rect(DrawRect(rect, res, total_max_abs if total_max_abs > 0 else 1e-100))

                if sub_rect is not None:
                    for item in sub_rect:
                        self.calc_queue.append(CalcJob(item[0], item[1], 0.0))

                idx += 1

            del self.calc_queue[:idx]

            if sub_max_abs > total_max_abs:
                total_max_abs = sub_max_abs
                self.draw_mgr.change_max_abs(total_max_abs)

    def run(self):
        self.loop()
