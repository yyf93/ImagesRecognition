import cv2
import numpy as np
import common
import params
import face_recognition
from sklearn.model_selection import train_test_split
import image_extract_features

'''
用各个模型训练人脸的分类
'''

def accuracy_cv2_score(recognizer, X_test, y_test):
    '''
    对CV2模型LBPHFaceRecognizer_create进行评估
    Model Accuracy: 0.7586206896551724
    '''
    y_preds = []
    accurate = 0
    for face in X_test:
        y_preds.append(recognizer.predict(face)[0])
    for i in range(len(y_preds)):
        if np.array_equal(y_preds[i], y_test[i]):
            accurate += 1
    accuracy = accurate / len(y_test)
    print("Model Accuracy:", accuracy)


def train_faces_cv2_model(images):
    '''
    训练cv2分类模型并评估
    '''
    new_images = []
    new_labels = []
    # 训练AdaBoost分类器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    for img_path in images:
        img_path = img_path.replace('\\', '/')
        dir = img_path.split('/')[-2]
        print(f'{img_path} -- {dir}')
        face = image_extract_features.extract_face_haarcascade_features(img_path)
        if face is not None:
            new_images.append(face)
            label_read = common.reverseDict(params.face_label_mapping)[dir]
            new_labels.append(label_read)
        else:
            print(f'can\'t get {img_path} face...')
    np_images = np.array(new_images, dtype='object')
    np_labels = np.array(new_labels, dtype=np.int32)
    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(np_images, np_labels, test_size=0.2, random_state=42)
    recognizer.train(X_train, y_train)
    recognizer.save('./models/face_cv2_lbphface_sift.xml')
    # 评估模型
    accuracy_cv2_score(recognizer, X_test, y_test)


def test_faces_cv2_model(img_path):
    '''
    测试CV2分类模型
    '''
    # # 训练AdaBoost分类器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./models/face_cv2_lbphface_sift.xml')
    # 测试模型
    face = image_extract_features.extract_face_haarcascade_features(img_path)
    if face is not None:
        label, confidence = recognizer.predict(face)
        print(f'Label: {label} | Confidence: {confidence}')
    else:
        print(f'can\'t get {img_path} face...')


def train_faces_face_recognition_model(images):
    '''
    通过face_recognition训练人脸分类器
    '''
    # 加载所有人脸图片并获取编码
    known_faces_encodings = []
    known_faces_names = []

    for img_path in images:
        img_path = img_path.replace('\\', '/')
        dir = img_path.split('/')[-2]
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
        # 保存训练好的人脸分类器变量
        np.save("./models/face_face_recognition_faces_encodings.npy", known_faces_encodings)
        np.save("./models/face_face_recognition_faces_names.npy", known_faces_names)
        # 评估模型 TODO


def test_faces_face_recognition_model(img_path):
    '''
    通过face_recognition测试人脸分类器
    '''
    # 加载训练好的人脸分类器
    known_faces_encodings = np.load("./models/face_face_recognition_faces_encodings.npy")
    known_faces_names = np.load("./models/face_face_recognition_faces_names.npy")

    # 预测未知人脸
    unknown_face = face_recognition.load_image_file(img_path)
    unknown_face_locations = face_recognition.face_locations(unknown_face)
    print(f"Detected {len(unknown_face_locations)} faces.")
    if len(unknown_face_locations) > 0:
        unknown_encoding = face_recognition.face_encodings(unknown_face, unknown_face_locations)[0]
        # 这方法不好用 results = face_recognition.compare_faces(known_faces_encodings, unknown_encoding)
        # 计算未知人脸与训练集中人脸的距离
        distances = face_recognition.face_distance(known_faces_encodings, unknown_encoding)
        # 设置阈值
        threshold = 0.5
        # 找到距离最小的人脸
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
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
    train_faces_face_recognition_model(images)
    # testFacesFaceRegonModels('./images/test_faces/yyf_0.jpg')
    train_faces_cv2_model(images)
    # testFacesCV2Models('./images/test_faces/yyf_0.jpg')
