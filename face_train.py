import cv2
import numpy as np
import common
import params
import face_recognition
from sklearn.model_selection import train_test_split
import image_extract_features
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import image_extract_features
from face import *
import image_enhance

'''
用各个模型训练人脸的分类
'''
def train_faces(is_full_train=False):
    """
    使用resnet152去训练人脸
    """
    # 获取图书的配置信息
    face_tool = FaceTool()

    # 训练集路径
    train_dir = face_tool.face_train_dir

    # 对训练路径的人脸进行图像增强
    image_enhance.generate_face_images(train_dir)

    #是否全量训练
    if is_full_train:
        # 全量训练
        # 读取图像数据集
        images = common.get_trains_sub_folder_files(train_dir, [])
        print("全量训练 共计 %s 数据" % len(images))
        train_faces_resnet_model(face_tool, images, is_full_train)
    else:
        #增量训练
        # 判断是否有新增的图书图像
        images = face_tool.get_increment_train_images()
        print("增量训练 共计 %s 数据" % len(images))
        # 看是否需要增量训练模型
        if len(images) > 0:
            train_faces_resnet_model(face_tool, images, is_full_train)
        else:
            print('未有新增人脸素材， 不进行训练')


def train_faces_resnet_model(face_tool, images, is_full_train):
    '''
    加载预训练的ResNet152层模型
    '''
    resnet = models.resnet152(pretrained=True)
    resnet.fc = nn.Identity()  # 去除最后的全连接层

    # 提取训练集中的特征/ 如果全量训练则为空
    train_features = []
    train_labels = []

    if not is_full_train:
        # 如果增量训练， 加载模型， 加载特征和标签
        if common.is_exit_file(params.model_torch_faces_resnet):
            info_dict = torch.load(params.model_torch_faces_resnet)
            resnet_dict = info_dict["model_state_dict"]

            resnet.load_state_dict(resnet_dict)

            # 冻结前面的层（可选，根据需要解冻或冻结）
            if not is_full_train:
                for param in resnet.parameters():
                    param.requires_grad = False

            # 提取增量训练集中的特征
            train_features = info_dict["image_feautres_list"]
            train_labels = info_dict["image_labels_list"]

    # 对图像进行提取特征+标签
    for image_path in images:
        image_path = image_path.replace('\\', '/')
        uniq_id = image_path.split('/')[-2]
        print(f'----- path: {image_path} ------ uniq: {uniq_id}')
        # print(uniq_id)
        if uniq_id:
            train_labels.append(int(uniq_id))
            features = image_extract_features.extract_resnet_features(image_path, resnet)
            train_features.append(features)

    np_features = torch.cat(train_features, dim=0)
    np_labels = torch.tensor(train_labels)
    # 数据集划分 - 用于预测模型的准确率
    # X_train, X_test, y_train, y_test = train_test_split(np_features, np_labels, test_size=0.2, random_state=42)

    # num_classes为类别的数量
    num_classes = len(face_tool.all_face_infos)
    print(f'all classes nums: {num_classes}')

    # 构建分类器模型
    classifier = nn.Linear(np_features.shape[1], num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # 这里是全量进行训练
    X_train = np_features
    y_train = np_labels

    # 训练分类器
    num_epochs = 2000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Models_Evaluation.txt 只有当需要看准确率的时候才用上的代码
    # with torch.no_grad():
    #     outputs = classifier(X_test)
    #     _, predicted = torch.max(outputs, 1)
    #     accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    #     print(f"Validation Accuracy: {accuracy}")

    state_dict = {
        'model_state_dict': resnet.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        'image_feautres_list': train_features,
        'image_labels_list': train_labels
    }

    # 保存模型
    torch.save(state_dict, params.model_torch_faces_resnet)

    # 跟新配置文件中该图书已经训练过了
    face_tool.upadte_is_train_to_config(train_labels)



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


def train_faces_cv2_model(images, train_all=True):
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
        faces = image_extract_features.extract_face_haarcascade_features(img_path)
        if len(faces) > 0:
            for face in faces:
                new_images.append(face)
                label_read = common.reverseDict(params.face_label_mapping)[dir]
                new_labels.append(label_read)
        else:
            print(f'can\'t get {img_path} face...')
    np_images = np.array(new_images, dtype='object')
    np_labels = np.array(new_labels, dtype=np.int32)
    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(np_images, np_labels, test_size=0.2, random_state=42)
    if train_all:
        X_train = np_images
        y_train = np_labels
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
    faces = image_extract_features.extract_face_haarcascade_features(img_path)
    if len(faces) > 0 :
        for face in faces:
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
    # print(images)
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
    train_faces(is_full_train=False)
    # 读取图像数据集和标签
    # images = []
    # images = common.get_files_and_folder('./images/train_faces', images)
    # print("共计 %s 数据" % len(images))
    # train_faces_face_recognition_model(images)
    # train_faces_cv2_model(images)
    # testFacesFaceRegonModels('./images/test_faces/yyf_0.jpg')
    # testFacesCV2Models('./images/test_faces/yyf_0.jpg')
