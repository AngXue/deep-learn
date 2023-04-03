import math
import os
import os.path
import pickle

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def location_face(img):
    imgTemp = np.array(img)
    face_bound_boxes = face_recognition.face_locations(imgTemp)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for (top, right, bottom, left) in face_bound_boxes:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    del draw
    return img


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree'):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) == 1:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(img, knn_clf=None, model_path=None, distance_threshold=0.6):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_img = np.array(img)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img, predictions):
    face_landmarks_list = face_recognition.face_landmarks(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        fontText = ImageFont.truetype("font/simsun.ttc", 15, encoding="utf-8")
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=fontText)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            draw.line(face_landmarks[facial_feature], width=1)

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def predict_static(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def out_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    face_landmarks_list = face_recognition.face_landmarks(np.array(pil_image))
    draw = ImageDraw.Draw(pil_image)
    names = []

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        names.append(name)
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        fontText = ImageFont.truetype("font/simsun.ttc", 15, encoding="utf-8")  # 支持中文
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=fontText)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            draw.line(face_landmarks[facial_feature], width=1)

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    return pil_image, names


def save_pre(img_path, model_path, save_path):
    # 保存预测标注后的图片
    outImage, names = out_prediction_labels_on_image(img_path, predict_static(img_path, model_path=model_path))
    if not names:
        names.append("none")
    fileName = names[0] + '.' + os.path.basename(img_path)
    outImage.save(os.path.join(save_path, fileName))


def evaluate(save_path):
    allNum = os.listdir(save_path)
    if allNum == 0:
        return
    accNum = 0
    noneCnt = 0
    for root, ds, fs in os.walk(save_path):
        for f in fs:
            # 以 . 分割文件名， 比较前两段是否相等 例：test.test.png 中 test==test
            preName = os.path.splitext(f)[0].split('.')[0]
            realName = os.path.splitext(f)[0].split('.')[1]
            if preName == realName:
                accNum += 1
            elif preName == "none":
                noneCnt += 1
    return (accNum / len(allNum)), (noneCnt / len(allNum))
