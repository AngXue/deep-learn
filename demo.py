# coding:utf-8

import os
import sys
import time

import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog

import ui.MainWind as MainWind


# TODO: 加组件和模块化
# 打包: nuitka --standalone --show-memory --show-progress --output-dir=out --enable-plugin=pyqt5 --windows-disable-console --windows-icon-from-ico=pro.ico demo.py
class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = MainWind.Ui_MainWind(self)
        self.setWindowTitle("Main")
        self.cap = cv2.VideoCapture()  # 准备获取图像
        self.CAM_NUM = 0

    def showCamera(self):
        flag, self.image = self.cap.read()
        self.image = cv2.flip(self.image, 1)  # 左右翻转
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.ui.showCamerLabel.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.ui.showCamerLabel.setScaledContents(True)

    def openCamera(self):
        if not self.ui.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"请检测相机与电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.ui.timer_camera.start(30)

    def takePhoto(self):
        if self.ui.timer_camera.isActive():
            nowTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            # 如果没有res目录，则新建res目录
            if not os.path.exists('./res'):
                # 创建目录 res
                os.mkdir('./res')
            cv2.imwrite('./res/' + 'pic_' + str(nowTime) + '.png', self.image)
        else:
            self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                           "font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")

    def closeCamera(self):
        if self.ui.timer_camera.isActive():
            if self.cap.isOpened():
                self.cap.release()
            if self.ui.timer_camera.isActive():
                self.ui.timer_camera.stop()
        self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                       "font-size:28pt;\">摄像头已关闭</span><br/></p></body></html>")


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
