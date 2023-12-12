from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List

import numpy as np

if TYPE_CHECKING:
    pass


class DrawRect:
    def __init__(self, scr_rect, value, max_abs):
        self.scr_rect: List[int] = scr_rect
        self.value: float = 0
        self.color: np.ndarray = np.zeros(3, dtype=np.uint8)

        self.set_value(value, max_abs)

    def set_value(self, value, max_abs):
        self.value = value

        clipped_value = np.clip(value, -max_abs, max_abs)
        if clipped_value > 0:
            buf = 255 - int(255 * clipped_value / max_abs)
            self.color[0] = 255
            self.color[1] = buf
            self.color[2] = buf
        else:
            buf = 255 - int(255 * -clipped_value / max_abs)
            self.color[0] = buf
            self.color[1] = buf
            self.color[2] = 255


class DrawManager:
    def __init__(self):
        self.data: List[DrawRect] = []
        self.draw_queue: List[DrawRect] = []

    def add_rect(self, rect: DrawRect):
        self.data.append(rect)
        self.draw(rect)

    def change_max_abs(self, max_abs):
        for rect in self.data:
            rect.set_value(rect.value, max_abs)
            self.draw(rect)

    def draw(self, rect: DrawRect):
        self.draw_queue.append(rect)

    def draw_all(self):
        self.draw_queue += self.data

    def clear_data(self):
        del self.data
        self.data = []
