# coding:utf-8

import sys
import MainWind
import cv2
import time
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets


# TODO: 加组件和模块化
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
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            print(now_time)
            cv2.imwrite('../res/' + 'pic_' + str(now_time) + '.png', self.image)
            cv2.putText(self.image, 'The picture have saved !',
                        (int(self.image.shape[1] / 2 - 130), int(self.image.shape[0] / 2)),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        1.0, (255, 0, 0), 1)

            self.ui.timer_camera.stop()
            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 左右翻转
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.ui.showCamerLabel.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.showCamerLabel.setScaledContents(True)

    def closeCamera(self):
        if self.ui.timer_camera.isActive():
            ok = QtWidgets.QPushButton()
            cancel = QtWidgets.QPushButton()
            msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
            msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
            msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
            ok.setText(u'确定')
            cancel.setText(u'取消')
            if msg.exec_() != QtWidgets.QMessageBox.RejectRole:
                if self.cap.isOpened():
                    self.cap.release()
                if self.ui.timer_camera.isActive():
                    self.ui.timer_camera.stop()
                self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                               "font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
