# -- coding: utf-8 --
import warnings
import os
# 忽略警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from ui import Mywindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication,QMainWindow


if __name__ == '__main__':
    app = QApplication(argv)

    window = QMainWindow()
    ui = Mywindow(window)

    window.show()
    exit(app.exec_())
