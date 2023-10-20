# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pickle
import image_extract_features
from book import *
from face import *


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
    if image_extract_features.extract_face_from_img_to_file(tmp_path):
        return tmp_path
    else:
        return None


def predict_resnet_book(unknown_image_path):
    '''
    RESNET
    '''
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torch.nn.functional as F
    book_tool = BookTool()
    # 加载预训练的ResNet模型
    resnet = models.resnet152(pretrained=True)
    resnet.fc = nn.Identity()

    #加载模型和分类器
    info_dict = torch.load(params.model_torch_books_resnet)
    resnet_dict = info_dict["model_state_dict"]
    classifier_dict = info_dict["classifier_state_dict"]
    resnet.load_state_dict(resnet_dict, strict=False)

    # 提取特征
    unknown_features = image_extract_features.extract_resnet_features(unknown_image_path, resnet)

    #分类数量
    num_classes = len(book_tool.all_book_infos)
    classifier = nn.Linear(unknown_features.shape[1], num_classes)  # num_classes为类别的数量
    classifier.load_state_dict(classifier_dict)

    prediction = classifier(unknown_features)
    probabilities = F.softmax(prediction, dim=1)  # 应用Softmax函数获得概率分布
    predicted_label = torch.argmax(prediction).item()
    confidence = probabilities[0][predicted_label].item()  # 获取预测标签的置信度
    # 通过uniq_id 返回图书的内容
    isbn = common.reverse_dict(book_tool.current_book_isbn_id_json)[str(predicted_label)]
    print(f'predicted_label: {predicted_label} ------  isbn_uniq_id_dict: {book_tool.current_book_isbn_id_json}')
    return {isbn: confidence}


def predict_resnet_face(unknown_image_path):
    '''
    RESNET
    '''
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torch.nn.functional as F
    try:
        face_tool = FaceTool()
        # 加载预训练的ResNet模型
        resnet = models.resnet152(pretrained=True)
        resnet.fc = nn.Identity()

        # 加载模型和分类器
        info_dict = torch.load(params.model_torch_faces_resnet)
        resnet_dict = info_dict["model_state_dict"]
        classifier_dict = info_dict["classifier_state_dict"]
        resnet.load_state_dict(resnet_dict, strict=False)

        # 提取特征
        unknown_features = image_extract_features.extract_resnet_features(unknown_image_path, resnet)

        # 分类数量
        num_classes = len(face_tool.all_face_infos)
        # print(f'face all class nums: {num_classes}')
        classifier = nn.Linear(unknown_features.shape[1], num_classes)  # num_classes为类别的数量
        classifier.load_state_dict(classifier_dict)

        prediction = classifier(unknown_features)
        probabilities = F.softmax(prediction, dim=1)  # 应用Softmax函数获得概率分布
        predicted_label = torch.argmax(prediction).item()
        confidence = probabilities[0][predicted_label].item()  # 获取预测标签的置信度
        # 通过uniq_id 返回图书的内容
        face_name = face_tool.uniq_name_dict[str(predicted_label)]
        print(f'predicted_label: {predicted_label} ------  face_name: {face_name}')
        return json.dumps({"resnet_result": {face_name: confidence}}, ensure_ascii=False), ""
    except Exception as e:
        return None, str(e)


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
    try:
        model_features = np.array(image_extract_features.extract_opencv_features(img_path)).astype(np.float32)
        model_pred = sklearn_model.predict([model_features])
        results[model_type] = params.book_label_mapping[str(model_pred[0])]
    except Exception as e:
        results[model_type] = ''


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
        print(f'{img_path} can\'t get face')
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
    #sklearn_models = ['svm']
    if json_data is not None:
        #sklearn_result = json_data['sklearn_result']
        resnet_result = json_data['resnet_result']
        for k, v in resnet_result.items():
            results.append(k)
        #for k, v in sklearn_result.items():
            #if k in sklearn_models:
                #results.append(v)
    return list(set(results))


def get_face(json_data):
    '''
    根据服务端返回的json解析最可能的人脸是谁
    '''
    results = []
    if json_data is not None:
        face_resnet_label_result = json_data['resnet_result']
        # face_recognition_label_result = json_data['face_recognition_label_result']
        for k, v in face_resnet_label_result.items():
            results.append(k)
    return list(set(results))


def get_borrow_return_book_message(img_path, book_type, book_name):
    """
    借还书JSON生成
    """
    book_tool = BookTool()
    return book_tool.get_borrow_return_json_file(img_path, book_type, book_name)


def get_book_respone(json_str, content=""):
    """
    将server book信息返回给client
    """
    if json_str is None:
        json_str = "{}"
    message_json = json.loads(json_str)
    if "imgBytes" in message_json.keys():
        message_json.pop("imgBytes")
    print(message_json)
    if content == "":
        message_json["is_success"] = "True"
        message_json["error_message"] = content
    else:
        message_json["is_success"] = "False"
        message_json["error_message"] = content
    return message_json


def get_face_respone(json_str, content=""):
    """
    将server face信息返回给client
    """
    if json_str is None:
        json_str = "{}"
    message_json = json.loads(json_str)
    print(message_json)
    if content == "":
        message_json["is_success"] = "True"
        message_json["error_message"] = content
    else:
        message_json["is_success"] = "False"
        message_json["error_message"] = content
    return message_json


def get_add_book_message(img_path, isbn):
    """
    返回新增书籍的json数据
    """
    book_tool = BookTool()
    if book_tool.is_exit_isbn(isbn):
        print("已经存在这个isbn， 不需要新增")
        return None
    else:
        return book_tool.get_add_json_file(img_path, isbn)


if __name__ == '__main__':
    #print(predict_face_recongnition('./images/test_faces/lj_01.jpg'))
    img_file = './images/test_books/barcode4.jpg'
    #img_file = './images/train_books/9787506394314/9787506394314_angle60.jpg'
    #print(predict_resnet_book(img_file))
    #print(get_add_book_message(img_file, None))
    face_file = './images/test_faces/pb_0.jpg'
    pre_get_face(face_file)
    print(predict_resnet_face(face_file))
    #print(predict_resnet_face(face_file))
    #send_email(img_file='./tmp/barcode.jpg', action='borrow', book_id='111', book_name='test')