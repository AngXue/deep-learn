# coding:utf-8

import os
import sys

import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog

import face_recognition_knn as frknn
import ui.MainWind as MainWind

MODELPATH = 'res/trainer/trained_knn_model.clf'
PHOTOPATH = 'res/trainPhotos/'
TESTPATH = 'res/testPhotos/'
SAVEPATH = 'res/savePhotos/'


# 打包前请将faceRecognize.py和faceTrain.py移动到demo.py同级目录下
# 打包: nuitka --standalone --follow-imports --include-data-dir=res=res --mingw64 --show-memory --show-progress --output-dir=out --enable-plugin=pyqt5 --windows-disable-console --windows-icon-from-ico=pro.ico demo.py
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


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
        self.image = self.recognize(self.image)  # 返回识别结果
        # self.image = frknn.location_face(self.image)  # 仅框框
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
        # 关闭摄像头计时器
        self.closeCamera()
        # 如果没有res目录，则新建res目录
        name = self.nameInput()
        photoPath = PHOTOPATH + name + '/'
        if not os.path.exists(photoPath):
            # 创建目录 res
            os.mkdir(photoPath)
        _id = len(os.listdir(photoPath))
        self.cap.open(self.CAM_NUM)
        flag, saveImg = self.cap.read()
        saveImg = cv2.flip(saveImg, 1)  # 左右翻转
        cv2.imencode('.jpg', saveImg)[1].tofile(photoPath + str(_id + 1) + '.jpg')
        self.ui.timer_camera.start(30)
        # self.train()
        self.openCamera()

    def closeCamera(self):
        if self.ui.timer_camera.isActive():
            if self.cap.isOpened():
                self.cap.release()
            if self.ui.timer_camera.isActive():
                self.ui.timer_camera.stop()
        self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                       "font-size:28pt;\">摄像头已关闭</span><br/></p></body></html>")

    def recognize(self, img):
        if not os.path.exists(MODELPATH):
            self.train()
        return frknn.show_prediction_labels_on_image(img, frknn.predict(img, model_path=MODELPATH))

    def train(self):
        self.closeCamera()
        if not os.path.exists(PHOTOPATH):
            os.mkdir(PHOTOPATH)
        if len(os.listdir(PHOTOPATH)) == 0:
            self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                           "font-size:28pt;\">请先录入人脸</span><br/></p></body></html>")
            return
        self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                       "font-size:28pt;\">训练中</span><br/></p></body></html>")
        frknn.train(PHOTOPATH, model_save_path=MODELPATH, n_neighbors=1)
        self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                       "font-size:28pt;\">训练完毕</span><br/></p></body></html>")

    def nameInput(self):
        # 获取输入的姓名
        name = self.ui.nameInput.text()
        return name

    def batchTest(self):
        # 批量测试
        self.closeCamera()
        if not os.path.exists(TESTPATH):
            os.mkdir(TESTPATH)
        testNum = len(os.listdir(TESTPATH))
        if testNum == 0:
            self.ui.showCamerLabel.setText("<html><head/><body><p align=\"center\"><span style=\" "
                                           "font-size:28pt;\">请先录入测试人脸</span><br/></p></body></html>")
            return
        leftNum = testNum
        testPaths = findAllFile(TESTPATH)
        while leftNum > 0:
            self.ui.showCamerLabel.setText("测试中 " + str((testNum - leftNum) / testNum * 100) + "%")
            frknn.save_pre(next(testPaths), model_path=MODELPATH, save_path=SAVEPATH)
            leftNum -= 1
        accuracy, rejectionRate = frknn.evaluate(SAVEPATH)
        self.ui.showCamerLabel.setText("测试完毕 正确率: " + str(accuracy * 100) + "% | 拒识率: " + str(rejectionRate * 100) + "%"
                                       + "\n测试结果保存在: " + os.path.join(os.getcwd(), 'res\\savePhotos'))


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
