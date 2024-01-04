import random
import time
import threading

import numpy as np

from Viewer import Viewer
from Charge import LineChargeDist
from DrawManager import DrawManager, DrawRect
from CalcManager import CalcManager


def run():
    exit_flag = [False]

    draw_mgr = DrawManager()
    calc_mgr = CalcManager(draw_mgr, exit_flag)

    charge_depth = 1e-1
    charge_list = [
        LineChargeDist(-0.1, 0.05, 0.1, 0.05,density=1e-9, depth=charge_depth),
        LineChargeDist(-0.1, -0.05, 0.1, -0.05,density=-1e-9, depth=charge_depth),
        # LineChargeDist(-0.2, 0.2, -0.2, -0.2,density=1e-9, depth=charge_depth),
        # LineChargeDist(0.2, 0.2, 0.2, -0.2, density=-1e-9, depth=charge_depth)
    ]
    for c in charge_list:
        calc_mgr.add_charge(c)

    calc_th = threading.Thread(target=calc_mgr.run)
    calc_th.start()

    viewer = Viewer('test', 1000, 800, draw_mgr, calc_mgr, exit_flag)
    viewer.run_gui()

    time.sleep(0.5)


if __name__ == '__main__':
    run()
