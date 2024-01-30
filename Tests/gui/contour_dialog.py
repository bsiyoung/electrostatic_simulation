from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class ContourDialog(QDialog):
    def __init__(self, communicate):
        super().__init__()
        self.communicate = communicate
        self.setWindowTitle('Set contour')
        self.initUI()
    def initUI(self):
        layout = QHBoxLayout()

        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0, 99)
        layout.addWidget(QLabel('scale'))
        layout.addWidget(self.scale_spinbox)

        self.btn_ok = QPushButton('확인')
        self.btn_ok.clicked.connect(self.accept)
        layout.addWidget(self.btn_ok)

        self.setLayout(layout)

    # 확인 버튼 누르면 저장하는 함수
    def accept(self):
        self.communicate.contour_ok_signal.emit(self.scale_spinbox.value())
        super().accept()