import random
import time
import threading

from Viewer import Viewer
from DrawManager import DrawManager, DrawRect
from CalcManager import CalcManager


def run():
    exit_flag = [False]

    draw_mgr = DrawManager()

    calc_mgr = CalcManager(draw_mgr, exit_flag)
    calc_th = threading.Thread(target=calc_mgr.run)
    calc_th.start()

    viewer = Viewer('test', 1000, 800, draw_mgr, calc_mgr, exit_flag)
    viewer.run_gui()

    time.sleep(0.5)


if __name__ == '__main__':
    run()
