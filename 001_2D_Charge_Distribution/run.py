import threading
import time

from Viewer import Viewer
from DrawManager import DrawManager, DrawRect


def run():
    exit_flag = [False]

    draw_mgr = DrawManager()

    sz = 4
    for i in range(10, 600, sz):
        for j in range(10, 600, sz):
            draw_mgr.add_rect(DrawRect((i, j, i + sz, j + sz), 10, 30))
    viewer = Viewer('test', 1000, 800, draw_mgr, exit_flag)

    viewer.run_gui()


if __name__ == '__main__':
    run()
