from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List

if TYPE_CHECKING:
    pass


class DrawRect:
    def __init__(self, scr_rect, value, max_abs):
        self.scr_rect: List[int] = scr_rect
        self.value: float = 0
        self.color: str = '#FFFFFF'

        self.set_value(value, max_abs)

    def set_value(self, value, max_abs):
        self.value = value
        if value > 0:
            buf = 255 - int(255 * value / max_abs)
            self.color = self.rgb_to_hex(255, buf, buf)
        else:
            buf = 255 - int(255 * -value / max_abs)
            self.color = self.rgb_to_hex(buf, buf, 255)

    @staticmethod
    def rgb_to_hex(r, g, b):
        return '#%02X%02X%02X' % (r, g, b)


class DrawManager:
    def __init__(self):
        self.data: List[DrawRect] = []
        self.draw_queue: List[DrawRect] = []

    def add_rect(self, rect: DrawRect):
        self.data.append(rect)
        self.draw(rect)

    def draw(self, rect: DrawRect):
        self.draw_queue.append(rect)

    def draw_all(self):
        self.draw_queue += self.data

    def clear_data(self):
        del self.data
        self.data = []
