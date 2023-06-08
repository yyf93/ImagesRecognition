import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import params
import image_extract_features

'''
serverm client用的api
'''

def pre_get_book(tmp_path):
    '''
    预处理
    '''
    return True, tmp_path

def pre_get_face(tmp_path):
    '''
    预处理
    '''
    return True, tmp_path


def predict_resnet(unknown_image_path):
    '''
    RESNET
    '''
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torch.nn.functional as F
    # 加载预训练的ResNet模型
    resnet = models.resnet152(pretrained=True)
    resnet.fc = nn.Identity()  # 去除最后的全连接层

    unknown_features = image_extract_features.extract_resnet_features(unknown_image_path, resnet)

    num_classes = len(params.book_label_mapping)
    # # 构建分类器模型
    classifier = nn.Linear(unknown_features.shape[1], num_classes)  # num_classes为类别的数量
    #加载模型
    classifier.load_state_dict(torch.load(params.model_books_torch_resnet))

    prediction = classifier(unknown_features)
    probabilities = F.softmax(prediction, dim=1)  # 应用Softmax函数获得概率分布
    predicted_label = torch.argmax(prediction).item()
    confidence = probabilities[0][predicted_label].item()  # 获取预测标签的置信度

    return params.book_label_mapping[str(predicted_label)], confidence


def predict_knn_sklearn(img_path, features=100, features_per=0.9):
    '''
    knn sklearn
    '''
    knn = KNeighborsClassifier(n_neighbors=len(set(params.book_label_mapping.keys())), metric='euclidean')
    knn_features = np.array(image_extract_features.extract_opencv_features(img_path, features, features_per)).astype(np.float32)
    knn_pred = knn.predict(knn_features)
    return knn_pred, 0


def predict_svm_sklearn(img_path, features=100, features_per=0.9):
    '''
    svm sklearn
    '''
    svm = SVC(kernel='linear', C=1.0)
    svm_features = np.array(image_extract_features.extract_opencv_features(img_path, features, features_per)).astype(
        np.float32)
    svm_pred = svm.predict(svm_features)
    return svm_pred, 0


def predict_cv2_lbphface(img_path):
    '''
    cv2 lbph face
    '''
    #Model
    model_lbphface = cv2.face.LBPHFaceRecognizer_create()
    ## LBPHFACE
    model_lbphface.read(params.model_faces_cv2_lbphface)
    test_img = cv2.imread(img_path)
    face = image_extract_features.extract_face_haarcascade_features(test_img)
    if face is not None:
        label, confidence = model_lbphface.predict(face)
        return params.face_label_mapping[str(label)], confidence
    else:
        print(f'can\'t get face')
        return 'Unknown', 0


def predict_face_recongnition(img_path):
    '''
    face_recognition
    '''
    import face_recognition

    ## face_recognition
    known_faces_encodings = np.load(params.params_face_recognition_faces_encoding)
    known_faces_names = np.load(params.params_face_recognition_faces_names)
    # 预测未知人脸
    unknown_face = face_recognition.load_image_file(img_path)
    unknown_face_locations = face_recognition.face_locations(unknown_face)
    if len(unknown_face_locations) > 0:
        unknown_encoding = face_recognition.face_encodings(unknown_face, unknown_face_locations)[0]
        # 计算未知人脸与训练集中人脸的距离
        distances = face_recognition.face_distance(known_faces_encodings, unknown_encoding)
        # 设置阈值
        # threshold = 0.5
        # 找到距离最小的人脸
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        # print()
        # if min_distance <= threshold:
            # 识别为训练集中的人
        predicted_label = known_faces_names[min_distance_index]
        # else:
        #     # 不属于训练集中的任何一个人
        #     predicted_label = "Unknown"
        return predicted_label, min_distance
    else:
        return 'Unknown', 0


def getMaxConfidenceBooks(json_data):
    results = []
    if json_data is not None:
        svm_result = json_data['svm_result']
        knn_result = json_data['knn_result']
        knn_confidence = json_data['knn_confidence']
        resnet_result = json_data['resnet_result']
        resnet_confidence = json_data['resnet_confidence']
        results.append(svm_result)
        # if float(knn_confidence) > 0.2:
        results.append(knn_result)
        # if float(resnet_confidence) > 0.4:
        results.append(resnet_result)
    return list(set(results))


def getMaxConfidenceFaces(json_data):
    results = []
    if json_data is not None:
        face_recognition_label_result = json_data['face_recognition_label_result']
        face_recognition_confidence = json_data['face_recognition_confidence']
        lbphface_result = json_data['lbphface_result']
        lbphface_confidence = json_data['lbphface_confidence']
        # if float(face_recognition_confidence) > 0.3:
        results.append(face_recognition_label_result)
        # if float(lbphface_confidence) > 10:
        results.append(lbphface_result)
    return list(set(results))


if __name__ == '__main__':
    print('test')