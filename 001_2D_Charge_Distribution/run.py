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

    '''
    sz = 1
    for i in range(10, 700, sz):
        for j in range(10, 900, sz):
            draw_mgr.add_rect(DrawRect((i, j, i + sz - 1, j + sz - 1), random.randint(-10, 10), 20))
    '''

    calc_th = threading.Thread(target=calc_mgr.run)
    calc_th.start()

    viewer = Viewer('test', 1000, 800, draw_mgr, calc_mgr, exit_flag)
    viewer.run_gui()

    time.sleep(0.5)


if __name__ == '__main__':
    run()
