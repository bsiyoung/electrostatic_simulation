from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ColorDialog(QDialog):
    def __init__(self, communicate):
        super().__init__()
        self.communicate = communicate
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
        min = tuple(spinbox.value() for spinbox in self.spinboxes['min'])
        max = tuple(spinbox.value() for spinbox in self.spinboxes['max'])
        ref = tuple(spinbox.value() for spinbox in self.spinboxes['ref'])
        self.communicate.color_ok_signal.emit([min, max, ref])
        super().accept()