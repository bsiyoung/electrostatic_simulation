import sys

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap
from PyQt5.QtCore import Qt


class PlotWindow(QWidget):
    def __init__(self, data, pts, plot_size, window_size, grid_size, img_file):
        super().__init__()
        self.init_ui(window_size)

        self.min_value = data['min']
        self.max_value = data['max']
        self.data = data['data']
        self.pts = pts
        self.plot_size = plot_size
        self.window_size = window_size
        self.grid_size = grid_size

        self.img_file = img_file

    def init_ui(self, window_size):
        self.setGeometry(0, 0, window_size, window_size)
        self.setWindowTitle('001. Electric Potential of Line Charge Distribution (2D)')

        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw_data(qp)
        self.draw_axis(qp)
        self.draw_distribution(qp)
        qp.end()

    def draw_data(self, qp):
        qp.setPen(QPen(Qt.NoPen))

        px_per_grid = self.window_size * 0.5 / (self.plot_size / self.grid_size)
        half_window = self.window_size * 0.5

        mid_value = (self.max_value + self.min_value) / 2
        half_range = mid_value - self.min_value
        for idx, (grid_pos, val) in enumerate(self.data):
            if val < mid_value:
                c = (val - self.min_value) / half_range * 255
                qp.setBrush(QColor(0, int(c), int(255 - c)))
            else:
                c = (val - mid_value) / half_range * 255
                qp.setBrush(QColor(int(c), int(255 - c), 0))

            x0 = int(half_window + grid_pos[0] * px_per_grid)
            y0 = int(half_window + (-grid_pos[1]) * px_per_grid)
            qp.drawRect(x0, y0, int(px_per_grid) + 1, int(px_per_grid) + 1)

            if (idx + 1) % 100 == 0 or idx == (len(self.data) - 1):
                perc = round((idx + 1) / len(self.data) * 100, 3)
                print('\rDrawing... {}/{} ({}%)'.format(idx + 1, len(self.data), perc), end='')
        print()

    def draw_axis(self, qp):
        half_window = int(self.window_size / 2)
        qp.setPen(QPen(QColor(100, 100, 100), 1))

        # X axis
        qp.drawLine(0, half_window, self.window_size, half_window)
        # Y axis
        qp.drawLine(half_window, 0, half_window, self.window_size)

    def draw_distribution(self, qp):
        qp.setPen(QPen(QColor(0, 0, 0), 5))
        pixel_per_meter = self.window_size / (self.plot_size * 2)
        half_window = int(self.window_size / 2)

        x0 = int(half_window + pixel_per_meter * self.pts[0][0])
        y0 = int(half_window - pixel_per_meter * self.pts[0][1])
        x1 = int(half_window + pixel_per_meter * self.pts[1][0])
        y1 = int(half_window - pixel_per_meter * self.pts[1][1])
        qp.drawLine(x0, y0, x1, y1)

    def mouseReleaseEvent(self, e):
        p = self.grab()
        if not p.isNull():
            p.setDevicePixelRatio(2)
            p.save(self.img_file, self.img_file.split('.')[-1], 100)


def draw_plot(data, pts, plot_size, window_size, grid_size, img_file):
    app = QApplication(sys.argv)
    ex = PlotWindow(data, pts, plot_size, window_size, grid_size, img_file)
    sys.exit(app.exec_())
