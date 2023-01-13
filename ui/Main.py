# coding:utf-8

import sys
import open
from PyQt5.QtWidgets import QApplication, QDialog
from raiseCamera.main import openCamera as mOpenCamera


# TODO: bug bug bug...
class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = open.Ui_Dialog()
        self.ui.setupUi(self)

    @staticmethod
    def openCamera():
        # mOpenCamera()
        print("pass")


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
