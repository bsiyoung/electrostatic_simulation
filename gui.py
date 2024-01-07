import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys, os

sim_conf = {}
phy_rect = (-0.4, 0.4, 0.4, -0.4)
mpp = 1e-4
down_sampling = 1
plots = {}
color = {}
min = (0, 0, 255)
max = (255, 0, 0)
ref = (255, 255, 255)
contour = {}
scale = 3
ref_point = None
device = 'gpu'
charges = []
class MyApp(QDialog):
    def __init__(self):
        super().__init__()
        self.data = {}
        self.initUI()

    def initUI(self):
        # Layout 생성
        self.main_layout = QFormLayout()

        # 파라미터들
        # phy_rect
        self.p_x1 = QDoubleSpinBox()
        self.p_y1 = QDoubleSpinBox()
        self.p_x2 = QDoubleSpinBox()
        self.p_y2 = QDoubleSpinBox()
        self.phy_rect_layout = QHBoxLayout()
        self.phy_rect_layout.addWidget(QLabel("x1: "))
        self.phy_rect_layout.addWidget(self.p_x1)
        self.phy_rect_layout.addWidget(QLabel("y1: "))
        self.phy_rect_layout.addWidget(self.p_y1)
        self.phy_rect_layout.addWidget(QLabel("x2: "))
        self.phy_rect_layout.addWidget(self.p_x2)
        self.phy_rect_layout.addWidget(QLabel("y2: "))
        self.phy_rect_layout.addWidget(self.p_y2)

        # mpp
        self.mpp_layout = QHBoxLayout()
        self.mpp_layout.addWidget(QLineEdit())

        # down_sampling
        self.down_sampling = 1

        # plots
        self.plots_layout = QHBoxLayout()
        self.btn_layout = QVBoxLayout()
        self.left_list = QListWidget()
        self.right_list = QListWidget()
        self.right_list.itemDoubleClicked.connect(self.open_dialog)
        self.left_list.addItems(['color', 'contour'])
        self.left_btn = QPushButton('←')
        self.left_btn.clicked.connect(self.move_left)
        self.right_btn = QPushButton('→')
        self.right_btn.clicked.connect(self.move_right)
        self.btn_layout.addWidget(self.right_btn)
        self.btn_layout.addWidget(self.left_btn)
        self.plots_layout.addWidget(self.left_list)
        self.plots_layout.addLayout(self.btn_layout)
        self.plots_layout.addWidget(self.right_list)

        # ref_point
        self.ref_point_layout = QHBoxLayout()
        self.r_x = QDoubleSpinBox()
        self.r_y = QDoubleSpinBox()
        self.inf = QCheckBox()
        self.ref_point_layout.addWidget(QLabel('x: '))
        self.ref_point_layout.addWidget(self.r_x)
        self.ref_point_layout.addWidget(QLabel('y: '))
        self.ref_point_layout.addWidget(self.r_y)
        self.ref_point_layout.addWidget(QLabel('inf: '))
        self.ref_point_layout.addWidget(self.inf)

        # device
        self.device_layout = QHBoxLayout()
        self.device_type = QComboBox()
        self.device_type.addItems(['cpu', 'gpu'])
        self.device_layout.addWidget(self.device_type)

        # charges
        self.charge_num = 0
        self.charges_layout = QHBoxLayout()
        self.charges_btn_layout = QHBoxLayout()
        self.append_btn = QPushButton('add')
        self.append_btn.clicked.connect(self.append_charge)
        self.delete_btn = QPushButton('del')
        self.delete_btn.clicked.connect(self.delete_charge)
        self.charges_btn_layout.addWidget(self.append_btn)
        self.charges_btn_layout.addWidget(self.delete_btn)
        self.charges_layout.addLayout(self.charges_btn_layout)
        self.charges_list = QListWidget()
        self.charges_layout.addWidget(self.charges_list)

        # buttons
        self.button_layout = QHBoxLayout()
        self.ok_btn = QPushButton('ok')
        self.button_layout.addWidget(self.ok_btn)

        self.main_layout.addRow('phy_rect', self.phy_rect_layout)
        self.main_layout.addRow('mpp', self.mpp_layout)
        self.main_layout.addRow('plots', self.plots_layout)
        self.main_layout.addRow('ref_point', self.ref_point_layout)
        self.main_layout.addRow('device', self.device_layout)
        self.main_layout.addRow('charges', self.charges_layout)
        self.main_layout.addRow('', self.button_layout)

        self.setLayout(self.main_layout)
        self.resize(500, 600)
        self.setWindowTitle('정전기 시뮬레이션')
        self.show()
    # plots에서 오른쪽 버튼 누르면 이동하는 함수
    def move_right(self):
        current_item = self.left_list.currentItem()
        if current_item:
            self.right_list.addItem(current_item.text())
            self.left_list.takeItem(self.left_list.row(current_item))
    # plots에서 왼쪽 버튼 누르면 이동하는 함수
    def move_left(self):
        current_item = self.right_list.currentItem()
        if current_item:
            self.left_list.addItem(current_item.text())
            self.right_list.takeItem(self.right_list.row(current_item))
    # 오른쪽 리스트에서 더블클릭 하면 새창 나오는 함수
    def open_dialog(self, item):
        if item.text() == 'color':
            dialog = ColorDialog()
        elif item.text() == 'contour':
            dialog = ContourDialog()
        dialog.exec_()

    # charge를 추가하는 함수
    def append_charge(self):
        self.charge_num += 1
        self.charges_list.addItem(f'charge{self.charge_num}')
    # charge를 삭제하는 함수
    def delete_charge(self):
        current_item = self.charges_list.currentItem()
        if current_item:
            self.charges_list.takeItem(self.charges_list.row(current_item))
    # 사용자 입력을 딕셔너리에 저장하는 함수

class ColorDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set color')
        self.initUI()
    def initUI(self):
        layout = QGridLayout()

        labels = ['min', 'max', 'ref']
        self.spinboxes = {label: [QSpinBox() for _ in range(3)] for label in labels}

        for i, label in enumerate(labels):
            layout.addWidget(QLabel(label), i, 0)
            for j in range(3):
                self.spinboxes[label][j].setRange(0, 255)
                layout.addWidget(self.spinboxes[label][j], i, j + 1)

        self.btn_ok = QPushButton('확인')
        self.btn_ok.clicked.connect(self.accept)
        layout.addWidget(self.btn_ok)

        self.setLayout(layout)

    # 확인 버튼 누르면 저장하는 함수
    def accept(self):
        global min, max, ref
        min = tuple(spinbox.value() for spinbox in self.spinboxes['min'])
        max = tuple(spinbox.value() for spinbox in self.spinboxes['max'])
        ref = tuple(spinbox.value() for spinbox in self.spinboxes['ref'])
        super().accept()

class ContourDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set contour')
        self.initUI()
    def initUI(self):
        layout = QHBoxLayout()

        self.scale_spinbox = QSpinBox()
        self.scale_spinbox.setRange(0, 99)
        layout.addWidget(QLabel('scale'))
        layout.addWidget(self.scale_spinbox)

        self.btn_ok = QPushButton('확인')
        self.btn_ok.clicked.connect(self.accept)
        layout.addWidget(self.btn_ok)

        self.setLayout(layout)

    # 확인 버튼 누르면 저장하는 함수
    def accept(self):
        global scale
        scale = self.scale_spinbox.value()
        super().accept()

class ChargeDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set charge')
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.values = []


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
