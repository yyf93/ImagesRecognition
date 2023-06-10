import cv2
import numpy as np
import pickle
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
    classifier.load_state_dict(torch.load(params.model_torch_books_resnet))

    prediction = classifier(unknown_features)
    probabilities = F.softmax(prediction, dim=1)  # 应用Softmax函数获得概率分布
    predicted_label = torch.argmax(prediction).item()
    confidence = probabilities[0][predicted_label].item()  # 获取预测标签的置信度

    return {params.book_label_mapping[str(predicted_label)]: confidence}


def predict_sklearn_books_models(img_path):
    '''
    decision_tree sklearn
    '''
    results = {}
    predict_common_sklearn(img_path, params.model_cv2_books_knn, results, 'knn')
    predict_common_sklearn(img_path, params.model_cv2_books_svm, results, 'svm')
    predict_common_sklearn(img_path, params.model_cv2_books_decision_tree, results, 'decision_tree')
    predict_common_sklearn(img_path, params.model_cv2_books_random_forest, results, 'random_forest')
    return results


def predict_common_sklearn(img_path, model_name, results, model_type):
    '''
    sklearn
    '''
    with open(model_name, 'rb') as file:
        sklearn_model = pickle.load(file)
    model_features = np.array(image_extract_features.extract_opencv_features(img_path)).astype(np.float32)
    model_pred = sklearn_model.predict([model_features])
    results[model_type] = params.book_label_mapping[str(model_pred[0])]


def predict_cv2_lbphface(img_path):
    '''
    cv2 lbph face
    '''
    results = {}
    #Model
    model_lbphface = cv2.face.LBPHFaceRecognizer_create()
    ## LBPHFACE
    model_lbphface.read(params.model_cv2_faces_lbphface)
    faces = image_extract_features.extract_face_haarcascade_features(img_path)
    if len(faces) > 0:
        for face in faces:
            label, confidence = model_lbphface.predict(face)
        results[params.face_label_mapping[str(label)]] = str(confidence)
        return results
    else:
        print(f'can\'t get face')
        return {'Unknown': '0'}


def predict_face_recongnition(img_path):
    '''
    face_recognition
    '''
    import face_recognition
    results = {}
    ## face_recognition
    known_faces_encodings = np.load(params.params_face_recognition_faces_encoding)
    known_faces_names = np.load(params.params_face_recognition_faces_names)
    # 预测未知人脸
    unknown_face = face_recognition.load_image_file(img_path)
    unknown_face_locations = face_recognition.face_locations(unknown_face)
    # print(len(unknown_face_locations))
    if len(unknown_face_locations) > 0:
        unknown_encodings = face_recognition.face_encodings(unknown_face, unknown_face_locations)
        for unknown_encoding in unknown_encodings:
            # 计算未知人脸与训练集中人脸的距离
            distances = face_recognition.face_distance(known_faces_encodings, unknown_encoding)
            # 找到距离最小的人脸
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            predicted_label = known_faces_names[min_distance_index]
            results[str(predicted_label)] = str(min_distance)
        return results
    else:
        return {'Unknown': '0'}


def get_book(json_data):
    '''
    根据服务端返回的json解析最可能的图书是谁
    '''
    results = []
    sklearn_models = ['svm', 'knn', 'decision_tree']
    if json_data is not None:
        sklearn_result = json_data['sklearn_result']
        resnet_result = json_data['resnet_result']
        for k, v in sklearn_result.items():
            if k in sklearn_models:
                results.append(v)
        for k, v in resnet_result.items():
            results.append(k)
    return list(set(results))


def get_face(json_data):
    '''
    根据服务端返回的json解析最可能的人脸是谁
    '''
    results = []
    if json_data is not None:
        face_recognition_label_result = json_data['face_recognition_label_result']
        for k, v in face_recognition_label_result.items():
            results.append(k)
    return list(set(results))


if __name__ == '__main__':
    print(predict_face_recongnition('./images/test_faces/lj_01.jpg'))
    print('test')