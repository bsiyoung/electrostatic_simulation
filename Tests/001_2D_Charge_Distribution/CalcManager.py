from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from typing import List, Tuple

import numpy as np
from DrawManager import DrawRect
from Charge import LineChargeDist

if TYPE_CHECKING:
    from DrawManager import DrawManager, DrawRect


class CalcJob:
    def __init__(self,
                 scr_rect: Tuple[int, int, int, int],
                 phy_rect: Tuple[float, float, float, float],
                 ref_voltage: float):
        self.scr_rect = scr_rect
        self.phy_rect = phy_rect

        self.t_scr_x, self.t_phy_x = self.__get_target_pos(scr_rect[0], scr_rect[2], phy_rect[0], phy_rect[2])
        self.t_scr_y, self.t_phy_y = self.__get_target_pos(scr_rect[1], scr_rect[3], phy_rect[1], phy_rect[3])

        scr_dx = scr_rect[2] - scr_rect[0]
        scr_dy = scr_rect[3] - scr_rect[1]
        phy_dx = phy_rect[2] - phy_rect[0]
        phy_dy = phy_rect[3] - phy_rect[1]
        if scr_dx != 0:
            self.mpp = phy_dx / scr_dx
        elif scr_dy != 0:
            self.mpp = phy_dy / scr_dy
        else:
            self.mpp = 0

        self.ref_voltage = ref_voltage

    @staticmethod
    def __get_target_pos(scr_st, scr_en, phy_st, phy_en):
        target_scr = (scr_st + scr_en) // 2
        target_phy = (phy_en + phy_st) / 2

        return target_scr, target_phy

    def do(self, charges: List[LineChargeDist]) -> Tuple[float, Tuple[int, int, int, int], Tuple]:
        res = -self.ref_voltage
        for charge in charges:
            res += charge.get_electric_potential(self.t_phy_x, self.t_phy_y)

        phy_border = (self.phy_rect[0] + self.mpp * (self.t_scr_x - self.scr_rect[0]),
                      self.phy_rect[1] - self.mpp * (self.t_scr_y - self.scr_rect[1]))

        sub_rect = (
            (  # Left Top
                (self.scr_rect[0], self.scr_rect[1], self.t_scr_x, self.t_scr_y),
                (self.phy_rect[0], self.phy_rect[1], phy_border[0], phy_border[1])
            ),
            (  # Right Top
                (self.t_scr_x + 1, self.scr_rect[1], self.scr_rect[2], self.t_scr_y),
                (phy_border[0] + self.mpp, self.phy_rect[1], self.phy_rect[2], phy_border[1])
            ) if self.scr_rect[0] != self.scr_rect[2] else None,
            (  # Left Bottom
                (self.scr_rect[0], self.t_scr_y + 1, self.t_scr_x, self.scr_rect[3]),
                (self.phy_rect[0], phy_border[1] - self.mpp, phy_border[0], self.phy_rect[3])
            ) if self.scr_rect[1] != self.scr_rect[3] else None,
            (  # Right Bottom
                (self.t_scr_x + 1, self.t_scr_y + 1, self.scr_rect[2], self.scr_rect[3]),
                (phy_border[0] + self.mpp, phy_border[1] - self.mpp, self.phy_rect[2], self.phy_rect[3])
            ) if (self.scr_rect[1] != self.scr_rect[3]) and (self.scr_rect[0] != self.scr_rect[2]) else None
        ) if (self.scr_rect[0] != self.scr_rect[2] or self.scr_rect[1] != self.scr_rect[3]) else None

        return res, self.scr_rect, sub_rect


class CalcManager:
    def __init__(self, draw_mgr: DrawManager, exit_flag: List[bool]):
        self.draw_mgr: DrawManager = draw_mgr
        self.exit_flag = exit_flag

        self.charges: List[LineChargeDist] = []
        self.calc_queue: List[Tuple[Tuple, DrawRect | None]] = []

        self.max_abs = 1e-100
        self.idx_threshold = 1

        self.lock = threading.Lock()

    def add_charge(self, charge: LineChargeDist):
        self.charges.append(charge)

    def get_inf_ref_electric_potential(self, x, y):
        res = 0
        for charge in self.charges:
            res += charge.get_electric_potential(x, y)

        return res

    def loop(self):
        while self.exit_flag[0] is False:
            time.sleep(0.001)
            self.lock.acquire()
            self.draw_mgr.lock.acquire()

            idx = 0
            max_abs_buf = self.max_abs
            if len(self.draw_mgr.draw_queue) > 1000 and self.idx_threshold > 2:
                self.idx_threshold -= 1
            elif self.idx_threshold < 300 and len(self.draw_mgr.draw_queue) < 100:
                self.idx_threshold += 1
            while idx < len(self.calc_queue) and idx < self.idx_threshold:
                if self.exit_flag[0] is True:
                    break

                # Draw new job
                calc_job = self.calc_queue[idx]

                # Do sub jobs of current job
                for job in calc_job[0]:
                    potential, target_rect, sub_rect = job.do(self.charges)

                    new_draw_data = DrawRect(target_rect, potential, self.max_abs)
                    self.draw_mgr.add_data(new_draw_data)
                    self.draw_mgr.draw(new_draw_data)

                    # Create new job
                    if sub_rect is not None:
                        new_sub_job = []
                        for r in sub_rect:
                            if r is None:
                                continue

                            new_sub_job.append(CalcJob(r[0], r[1], 0.0))

                        new_sub_job = tuple(new_sub_job)
                        self.calc_queue.append((new_sub_job, new_draw_data))

                    # Update max_abs value
                    if max_abs_buf < abs(potential):
                        max_abs_buf = abs(potential)

                # Remove parent data
                if calc_job[1] is not None:
                    self.draw_mgr.remove_data(calc_job[1])

                idx += 1

            del self.calc_queue[:idx]

            if self.max_abs < max_abs_buf:
                self.max_abs = max_abs_buf
                self.draw_mgr.change_max_abs(self.max_abs)

            self.draw_mgr.lock.release()
            self.lock.release()

    def run(self):
        self.loop()
