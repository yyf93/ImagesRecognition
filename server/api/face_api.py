from utils.face import *
from config import params
import numpy as np
from utils import common, image_extract_features
import cv2


def pre_get_face(tmp_path):
    '''
    预处理
    '''
    flag, imgs = image_extract_features.extract_face_from_img_to_file(tmp_path)
    if flag:
        return imgs
    else:
        return None


def predict_face_recongnition(img_path):
    '''
    face_recognition
    '''
    import face_recognition
    face_tool = FaceTool()
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
            # 通过uniq_id 返回图书的内容
            face_name = face_tool.uniq_name_dict[str(predicted_label)]
        return json.dumps({"face_recognition_result": {face_name: min_distance}}, ensure_ascii=False), ""
    else:
        return None, "在图像中未找到人脸"



def merge_face_recongnition_message(message, result_json):
    '''
    合并face_recongnition人脸识别所有的结果
    '''
    message = json.loads(message)
    result_json = json.loads(result_json)
    if message.get("face_recognition_result") and result_json.get("face_recognition_result") :
        result_list = result_json.get("face_recognition_result")
        result_list.append(common.get_key_from_json(message["face_recognition_result"])[0])
        result_json["face_recognition_result"] = list(set(result_list))
    else:
        values = common.get_key_from_json(message["face_recognition_result"])[0]
        result_json["face_recognition_result"] = [values]
    return json.dumps(result_json, ensure_ascii=False)



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

