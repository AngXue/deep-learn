import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# 名字标签
def name():
    names = []
    path = 'res/trainPhotos'
    if not os.path.exists(path):
        os.mkdir(path)
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    if len(imagePaths) != 0:
        for imagePath in imagePaths:
            _name = str(os.path.split(imagePath)[1].split('.', 2)[1])
            names.append(_name)
    return names


# 准备识别的图片
def face_detect_demo(img):
    names = name()
    # 加载训练数据集文件
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 加载数据
    if not os.path.exists('res/trainer/trainer.yml'):
        return img
    recognizer.read('res/trainer/trainer.yml')
    # 转化为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载分类器
    face_detector = cv2.CascadeClassifier('res/trainer/haarcascade_frontalface_alt2.xml')
    face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    # 将人脸框起来
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255))
        ids, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # confidence值大代表不可信
        if (confidence > 80) or (len(names) == 0):
            cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
        # 识别的是已经录入的人脸
        else:
            cv2.putText(img, ' ', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            fontText = ImageFont.truetype("font/simsun.ttc", 20, encoding="utf-8")
            draw.text((x + 15, y - 25), str(names[ids - 1]), (225, 0, 0), font=fontText)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img
