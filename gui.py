from queue import Queue

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os

from Tests.gui.color_dialog import ColorDialog
from Tests.gui.contour_dialog import ContourDialog
# from Tests.gui.sim_thread import SimulationThread
from Charge import ChargeDist
from Simulation import Simulation


class MyApp(QDialog):
    def __init__(self, communicate):
        super().__init__()
        self.communicate = communicate
        self.sim_conf = {}
        self.down_sampling = 1
        self.plots = {}
        self.color = {}
        self.min = (0, 0, 255)
        self.max = (255, 0, 0)
        self.ref = (255, 255, 255)
        self.contour = {}
        self.scale = 3
        self.ref_point = None
        self.device = 'gpu'
        self.charges = []
        self.initUI()

    def initUI(self):
        # Main Layout 생성
        self.main_layout = QFormLayout()

        # phy_rect
        self.setup_phy_rect()

        # mpp
        self.setup_mpp()

        # down_sampling
        self.down_sampling = 1

        # plots
        self.setup_plots()

        # ref_point
        self.setup_ref_point()

        # device
        self.setup_device()

        # charges
        self.setup_charges()

        # image title
        self.setup_img_title()

        # progress bar
        self.setup_progressbar()

        # buttons
        self.setup_ok_button()

        # layout setup
        self.setup_main_layouts()

        self.show()

    def setup_main_layouts(self):
        self.main_layout.addRow('phy_rect', self.phy_rect_layout)
        self.main_layout.addRow('mpp', self.mpp_layout)
        self.main_layout.addRow('plots', self.plots_layout)
        self.main_layout.addRow('ref_point', self.ref_point_layout)
        self.main_layout.addRow('device', self.device_layout)
        self.main_layout.addRow('charges', self.charges_layout)
        self.main_layout.addRow('title', self.title_layout)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addRow('', self.button_layout)
        self.setLayout(self.main_layout)
        self.resize(1000, 600)
        self.setWindowTitle('정전기 시뮬레이션')

    def setup_ok_button(self):
        self.button_layout = QHBoxLayout()
        self.ok_btn = QPushButton('ok')
        self.ok_btn.clicked.connect(self.make_input_dic)
        self.button_layout.addWidget(self.ok_btn)

    def setup_progressbar(self):
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

    def setup_charges(self):
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

    def setup_device(self):
        self.device_layout = QHBoxLayout()
        self.device_type = QComboBox()
        self.device_type.addItems(['cpu', 'gpu'])
        self.device_layout.addWidget(self.device_type)

    def setup_ref_point(self):
        self.ref_point_layout = QHBoxLayout()
        self.r_x = QDoubleSpinBox()
        self.r_y = QDoubleSpinBox()
        self.inf = QCheckBox()
        self.r_x.setRange(-100, 100)
        self.r_y.setRange(-100, 100)
        self.inf.stateChanged.connect(self.onCheckbox_changed)
        self.ref_point_layout.addWidget(QLabel('x: '))
        self.ref_point_layout.addWidget(self.r_x)
        self.ref_point_layout.addWidget(QLabel('y: '))
        self.ref_point_layout.addWidget(self.r_y)
        self.ref_point_layout.addWidget(QLabel('inf: '))
        self.ref_point_layout.addWidget(self.inf)

    def setup_plots(self):
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

    def setup_mpp(self):
        self.mpp_layout = QHBoxLayout()
        self.mpp_line = QLineEdit()
        self.mpp_layout.addWidget(self.mpp_line)

    def setup_phy_rect(self):
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
        self.p_x1.setRange(-100, 100)
        self.p_y1.setRange(-100, 100)
        self.p_x2.setRange(-100, 100)
        self.p_y2.setRange(-100, 100)

    def setup_img_title(self):
        self.title_layout = QHBoxLayout()
        self.title = QLineEdit()
        self.title_layout.addWidget(self.title)


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
            dialog = ColorDialog(self.communicate)
        elif item.text() == 'contour':
            dialog = ContourDialog(self.communicate)
        dialog.exec_()

    # charge를 추가하는 함수
    def append_charge(self):
        self.charge_num += 1
        name_list = ['x1', 'y1', 'x2', 'y2', 'density', 'depth']
        item = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel(f'charge{self.charge_num}'))
        for i in range(6):
            spin_box = QDoubleSpinBox()
            spin_box.setRange(-1e10, 1e10)
            spin_box.setDecimals(10)
            spin_box.setToolTip(name_list[i])
            layout.addWidget(spin_box)
        item.setLayout(layout)

        list_widget_item = QListWidgetItem()
        list_widget_item.setSizeHint(item.sizeHint())
        self.charges_list.addItem(list_widget_item)
        self.charges_list.setItemWidget(list_widget_item,item)

    # charge를 삭제하는 함수
    def delete_charge(self):
        current_item = self.charges_list.currentItem()
        if current_item:
            self.charges_list.takeItem(self.charges_list.row(current_item))

    # color의 데이터를 받으면 업데이트하는 함수
    def update_color(self, color):
        self.min = color[0]
        self.max = color[1]
        self.ref = color[2]

    # contour의 데이터를 받으면 업데이트하는 함수
    def update_contour(self, contour):
        self.contour = contour

    # ref_point의 체크박스의 체크유무 함수
    def onCheckbox_changed(self, state):
        if state == 2:
            self.r_x.setEnabled(False)
            self.r_y.setEnabled(False)
        else:
            self.r_x.setEnabled(True)
            self.r_y.setEnabled(True)

    # charges에 추가하는 함수
    def add_charge(self):
        item = QWidget()
        layout = QHBoxLayout()
        for _ in range(6):
            spinBox = QSpinBox()
            layout.addWidget(spinBox)
        item.setFixedHeight(50)
        item.setLayout(layout)
        self.charges_list.addItem(f'item{self.count}')
        self.count += 1
        self.charges_list.setItemWidget(self.charges_list.item(self.charges_list.count() - 1), item)

    # 사용자 입력을 딕셔너리에 저장하는 함수
    def make_input_dic(self):
        get_inputs(self)

    def get_phy_rect(self):
        return self.p_x1.value(), self.p_y1.value(), self.p_x2.value(), self.p_y2.value()

    def get_mpp(self):
        return eval(self.mpp_line.text())

    def get_down_sampling(self):
        return self.down_sampling

    def get_plots(self):
        selected_plots = [self.right_list.item(i).text() for i in range(self.right_list.count())]
        plots = {}
        for plot in selected_plots:
            if 'color' == plot:
                plots['potential_color'] = {
                    'min': self.min,
                    'max': self.max,
                    'ref': self.ref
                }
            if 'contour' == plot:
                plots['potential_contour'] = {
                    'scale': self.contour
                }
        return plots

    def get_ref_point(self):
        if self.inf.isChecked():
            return None
        else:
            return self.r_x.value(), self.r_y.value()

    def get_device(self):
        return self.device_type.currentText()

    def get_charges(self):
        charge_list = []
        for i in range(self.charges_list.count()):
            charge_item = self.charges_list.itemWidget(self.charges_list.item(i))
            charge_values = [charge_item.layout().itemAt(j).widget().value() for j in range(1, 7)]
            charge_list.append(eval(f'ChargeDist({charge_values[0]},'
                                    f'{charge_values[1]},'
                                    f'{charge_values[2]},'
                                    f'{charge_values[3]},'
                                    f'{charge_values[4]},'
                                    f'{charge_values[5]})'))
            # print(type(charge_values[0]))
        return charge_list

    def start_sim_thread(self, sim_conf):
        self.ok_btn.setEnabled(False)

        self.progress_q = Queue()

        self.thread = SimulationThread(sim_conf, self.progress_q)
        self.thread.finished.connect(self.show_complete_message)
        self.thread.finished.connect(self.timer_stop)
        self.thread.start()

        self.timer_start()

    def show_complete_message(self):
        self.ok_btn.setEnabled(True)
        QMessageBox.information(self,'Notice','이미지 생성완료.')

    def timer_start(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_queue)
        self.timer.start(100)  # Check every 100 milliseconds

    def timer_stop(self):
        self.timer.stop()

    def check_queue(self):
        while not self.progress_q.empty():
            value = self.progress_q.get()
            task = value['task']
            progress = round(value['progress'], 2)
            if value is not None:
                self.progress_bar.setValue(int(progress))
                self.progress_bar.setFormat(f'{task} : {progress}%')


class SimulationThread(QThread):
    sim_finished = pyqtSignal()

    def __init__(self, sim_conf, progress_q):
        super().__init__()
        self.sim_conf = sim_conf
        self.progress_q = progress_q

    def run(self):
        sim = Simulation(self.sim_conf)
        sim.run(self.progress_q)
        self.sim_finished.emit()

def get_inputs(ex):
    sim_conf = {
        'phy_rect': ex.get_phy_rect(),
        'mpp': ex.get_mpp(),
        'down_sampling': ex.get_down_sampling(),
        'plots': ex.get_plots(),
        'ref_point': ex.get_ref_point(),
        'device': ex.get_device(),
        'charges': ex.get_charges()
    }
    import pprint
    pprint.pprint(sim_conf, sort_dicts=False)
    ex.start_sim_thread(sim_conf)


class Communicate(QObject):
    color_ok_signal = pyqtSignal(list)
    contour_ok_signal = pyqtSignal(float)


def run():
    app = QApplication(sys.argv)
    communicate = Communicate()
    ex = MyApp(communicate)

    communicate.color_ok_signal.connect(ex.update_color)
    communicate.contour_ok_signal.connect(ex.update_contour)

    sys.exit(app.exec_())


if __name__ == '__main__':
    run()
