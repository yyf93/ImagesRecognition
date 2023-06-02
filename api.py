import cv2
import numpy as np
import params

# 加载预训练的人脸检测模型
# face_cascade = cv2.CascadeClassifier()

orb = cv2.ORB_create(nfeatures=100)
sift = cv2.SIFT_create(nfeatures=100)


########################## PRE BOOK
def pre_get_book(tmp_path):
    return True, tmp_path

########################## PRE BOOK
def pre_get_face(tmp_path):
    return True, tmp_path


######################RESNET
# 加载书籍图像并进行特征提取
def extract_features(image_path, model):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        features = model(input_batch)
    return features


def resnetBooks(unknown_image_path):
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torch.nn.functional as F
    # 加载预训练的ResNet模型
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()  # 去除最后的全连接层

    unknown_features = extract_features(unknown_image_path, resnet)

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

######################



######################KNN CV2
def knn(img_path):
    ## KNN
    model_knn = cv2.ml.KNearest_create()
    model_knn = model_knn.load(params.model_books_cv2_knn)

    test_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    kps, dess = sift.detectAndCompute(test_img, None)
    np_arra = np.array([dess[:90].reshape(-1)]).astype(np.float32)
    _, result, a, dists = model_knn.findNearest(np_arra, len(params.book_label_mapping) + 1)
    # 计算置信度
    confidences = 1.0 / (dists + 1e-6)  # 距离越小，置信度越高
    normalized_confidences = confidences / np.sum(confidences)  # 归一化置信度
    return params.book_label_mapping[str(_).split('.')[0]], np.average(normalized_confidences)

######################SVM CV2

def svm(img_path):
    ## SVM
    model_svm = cv2.ml.SVM_create()
    model_svm = model_svm.load(params.model_books_cv2_svm)

    test_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    kps, dess = sift.detectAndCompute(test_img, None)
    np_arra = np.array([dess[:90].reshape(-1)]).astype(np.float32)
    results = model_svm.predict(np_arra)
    return params.book_label_mapping[str(results[1][0][0]).split('.')[0]], results[0]


######################lbphface CV2
def getFaceFeatures(test_img):
    haar_face_cascade = cv2.CascadeClassifier(params.model_haar_face_cascade)

    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray, 1.3, 5)
    face = None
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
    return face

def lbphface(img_path):
    #Model
    model_lbphface = cv2.face.LBPHFaceRecognizer_create()
    ## LBPHFACE
    model_lbphface.read(params.model_faces_cv2_lbphface)

    test_img = cv2.imread(img_path)
    face = getFaceFeatures(test_img)
    if face is not None:
        label, confidence = model_lbphface.predict(face)
        # print(f'Label: {label} | Confidence: {confidence}')
        # if confidence <=50:
        #     return 'Unknown', 0
        return params.face_label_mapping[str(label)], confidence
    else:
        print(f'can\'t get face')
        return 'Unknown', 0

######################





######################face_recognition

def faceRecongnition(img_path):
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
        threshold = 0.5
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
######################





##########################
def getMaxConfidenceBooks(json_data):
    results = []
    if json_data is not None:
        svm_result = json_data['svm_result']
        knn_result = json_data['knn_result']
        knn_confidence = json_data['knn_confidence']
        resnet_result = json_data['resnet_result']
        resnet_confidence = json_data['resnet_confidence']
        results.append(svm_result)
        if float(knn_confidence) > 0.2:
            results.append(knn_result)
        if float(resnet_confidence) > 0.4:
            results.append(resnet_result)
    return list(set(results))


def getMaxConfidenceFaces(json_data):
    results = []
    if json_data is not None:
        face_recognition_label_result = json_data['face_recognition_label_result']
        face_recognition_confidence = json_data['face_recognition_confidence']
        lbphface_result = json_data['lbphface_result']
        lbphface_confidence = json_data['lbphface_confidence']
        if float(face_recognition_confidence) > 0.3:
            results.append(face_recognition_label_result)
        if float(lbphface_confidence) > 10:
            results.append(lbphface_result)
    return list(set(results))


if __name__ == '__main__':
    # print(knn('./images/test_books/manhuasuanfa_0.jpg'))
    # print(knn('./images/test_books/huobiyuweilai_0.jpg'))
    # print(svm('./images/test_books/manhuasuanfa_0.jpg'))
    # print(svm('./images/test_books/huobiyuweilai_0.jpg'))
    # print(resnetBooks('./images/test_books/huobiyuweilai_0.jpg'))
    # print(resnetBooks('./images/test_books/manhuasuanfa_0.jpg'))
    # print(lbphface('./images/test_faces/yyf_0.jpg'))
    # print(faceRecongnition('./images/test_faces/yyf_0.jpg'))
    print('test')