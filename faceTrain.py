import cv2
import numpy as np
import os
import common
import params
import face_recognition



def getFaceFeatures(img):
    face = None
    # Haar特征文件路径
    haar_face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    faces = haar_face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
    return face


def trainFacesCV2Models(images):
    new_images = []
    new_labels = []
    # # 训练AdaBoost分类器
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    for img_path in images:
        img_path = img_path.replace('\\', '/')
        dir = img_path.split('/')[-2]
        print(f'{img_path} -- {dir}')
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        face = getFaceFeatures(image)
        if face is not None:
            new_images.append(face)
            label_read = common.reverseDict(params.face_label_mapping)[dir]
            new_labels.append(label_read)
        else:
            print(f'can\'t get {img_path} face...')


    np_images = np.array(new_images, dtype='object')
    np_labels = np.array(new_labels, dtype=np.int32)

    recognizer.train(np_images, np_labels)
    recognizer.save('./models/face_cv2_lbphface_sift.xml')



def testFacesCV2Models(img_path):
    # # 训练AdaBoost分类器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./models/face_cv2_lbphface_sift.xml')

    # # # 测试模型
    test_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    face = getFaceFeatures(test_img)
    if face is not None:
        label, confidence = recognizer.predict(face)
        print(f'Label: {label} | Confidence: {confidence}')
    else:
        print(f'can\'t get {img_path} face...')


def trainFacesFaceRegonModels(images):

    # 加载所有人脸图片并获取编码
    known_faces_encodings = []
    known_faces_names = []

    for img_path in images:
        img_path = img_path.replace('\\', '/')
        dir = img_path.split('/')[-2]
        print(f'{img_path} -- {dir}')
        label = dir
        face = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(face)
        print(f"{img_path} --- Detected {len(face_locations)} faces.")
        if len(face_locations) > 0:
            encoding = face_recognition.face_encodings(face, face_locations)[0]
            known_faces_encodings.append(encoding)
            known_faces_names.append(label)
        else:
            print(f"{img_path} --- No face detected in the image.")

        # # 训练分类器
        # face_recognition.face_distance(known_faces, known_faces)
        # 步骤3：保存训练好的人脸分类器
        np.save("./models/face_face_recognition_faces_encodings.npy", known_faces_encodings)
        np.save("./models/face_face_recognition_faces_names.npy", known_faces_names)


def testFacesFaceRegonModels(img_path):
    # 步骤4：加载训练好的人脸分类器
    known_faces_encodings = np.load("./models/face_face_recognition_faces_encodings.npy")
    known_faces_names = np.load("./models/face_face_recognition_faces_names.npy")

    # 预测未知人脸
    unknown_face = face_recognition.load_image_file(img_path)
    unknown_face_locations = face_recognition.face_locations(unknown_face)
    print(f"Detected {len(unknown_face_locations)} faces.")
    if len(unknown_face_locations) > 0:
        unknown_encoding = face_recognition.face_encodings(unknown_face, unknown_face_locations)[0]
        # results = face_recognition.compare_faces(known_faces_encodings, unknown_encoding)
        # 计算未知人脸与训练集中人脸的距离
        distances = face_recognition.face_distance(known_faces_encodings, unknown_encoding)

        # 设置阈值
        threshold = 0.5

        # 找到距离最小的人脸
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        # print()
        if min_distance <= threshold:
            # 识别为训练集中的人
            predicted_label = known_faces_names[min_distance_index]
        else:
            # 不属于训练集中的任何一个人
            predicted_label = "Unknown"

        print(f"Predicted label: {predicted_label} -- {min_distance}")

    else:
        print("No face detected in the image.")


if __name__ == '__main__':
    # 读取图像数据集和标签
    images = []
    images = common.get_files_and_folder('./images/train_faces', images)
    print("共计 %s 数据" % len(images))
    trainFacesFaceRegonModels(images)
    # testFacesFaceRegonModels('./images/test_faces/yyf_0.jpg')
    trainFacesCV2Models(images)
    # testFacesCV2Models('./images/test_faces/yyf_0.jpg')
