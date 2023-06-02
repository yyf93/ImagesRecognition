import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import params
import common
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 加载书籍图像并进行特征提取
def extract_features(image_path, model):
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


def trainBooksTorchModels(images):
    # 加载预训练的ResNet模型
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()  # 去除最后的全连接层

    # 提取训练集中的特征
    train_features = []
    train_labels = []
    for image_path in images:
        image_path = image_path.replace('\\', '/')
        dir = image_path.split('/')[-2]
        train_labels.append(int(common.reverseDict(params.book_label_mapping)[dir]))
        features = extract_features(image_path, resnet)
        train_features.append(features)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.tensor(train_labels)
    # num_classes为类别的数量
    num_classes = len(params.book_label_mapping)
    # 构建分类器模型
    classifier = nn.Linear(train_features.shape[1], num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # 训练分类器
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # 保存模型
    torch.save(classifier.state_dict(), params.model_books_torch_resnet)





def trainBooksCV2Models(image_paths, features=100, features_per=0.9):
    orb = cv2.ORB_create(nfeatures=features)
    sift = cv2.SIFT_create(nfeatures=features)
    knn = cv2.ml.KNearest_create()
    svm = cv2.ml.SVM_create()

    new_sift_features = []
    new_labels = []
    get_feature_nums = int(features_per * features)
    dirs = []

    for oneImage in image_paths:
        oneImage = oneImage.replace('\\', '/')
        dir = oneImage.split('/')[-2]
        dirs.append(dir)
        try:
            image = cv2.cvtColor(cv2.imread(oneImage), cv2.COLOR_BGR2GRAY)
            kps, dess = sift.detectAndCompute(image, None)
            kpo, deso = orb.detectAndCompute(image, None)
            print(f'process ===={oneImage}=====', len(dess), len(deso), get_feature_nums)
            if len(dess) >= get_feature_nums:
                new_sift_features.append(dess[:get_feature_nums].reshape(-1))
                label_read = common.reverseDict(params.book_label_mapping)[dir]
                new_labels.append(label_read)
            else:
                print(f'{oneImage} can\'t get enough sift features from picture ')
        except Exception as e:
            print(f'Invalid image {oneImage}, skipping...')
            continue

    np_sift_images = np.array(new_sift_features).astype(np.float32)
    np_labels = np.array(new_labels).astype(np.int32)
    # 训练分类器
    # KNN Accuracy: 0.6363636363636364
    knn.train(np_sift_images, cv2.ml.ROW_SAMPLE, np_labels)
    knn.save('./models/books_cv2_knn_sift.xml')

    # SVM
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setC(10)
    # svm.setGamma(5)
    svm.train(np_sift_images, cv2.ml.ROW_SAMPLE, np_labels)
    svm.save('./models/books_cv2_svm_linear_sift.xml')



def trainBooksSklearnModels(image_paths, features=100, features_per=0.9):
    orb = cv2.ORB_create(nfeatures=features)
    sift = cv2.SIFT_create(nfeatures=features)
    new_sift_features = []
    new_labels = []
    get_feature_nums = int(features_per * features)
    dirs = []

    for oneImage in image_paths:
        oneImage = oneImage.replace('\\', '/')
        dir = oneImage.split('/')[-2]
        dirs.append(dir)
        try:
            image = cv2.cvtColor(cv2.imread(oneImage), cv2.COLOR_BGR2GRAY)
            kps, dess = sift.detectAndCompute(image, None)
            kpo, deso = orb.detectAndCompute(image, None)
            print(f'process ===={oneImage}=====', len(dess), len(deso), get_feature_nums)
            if len(dess) >= get_feature_nums:
                new_sift_features.append(dess[:get_feature_nums].reshape(-1))
                label_read = common.reverseDict(params.book_label_mapping)[dir]
                new_labels.append(label_read)
            else:
                print(f'{oneImage} can\'t get enough sift features from picture ')
        except Exception as e:
            print(f'Invalid image {oneImage}, skipping...')
            continue

    np_sift_images = np.array(new_sift_features).astype(np.float32)
    np_labels = np.array(new_labels).astype(np.int32)
    # 训练分类器
    # KNN Accuracy: 0.6363636363636364
    X_train, X_test, y_train, y_test = train_test_split(np_sift_images, np_labels, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=len(set(dirs)), metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("KNN Accuracy:", accuracy)
    with open('./models/books_sklearn_knn_sift.pkl', 'wb') as file:
        pickle.dump(knn, file)

    # # SVM rbf  C=1.0 Accuracy: 0.36363636363636365   linear  C=1.0 Accuracy: 0.8181818181818182  'poly', degree=3 C=1.0 Accuracy: 0.5454545454545454,, 'sigmoid', C=1.0 Accuracy: 0.36363636363636365
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Accuracy:", accuracy)
    with open('./models/books_sklearn_svm_linear_sift.pkl', 'wb') as file:
        pickle.dump(knn, file)




def testBooksSklearnModels(image_path, features=100, features_per=0.9):
    orb = cv2.ORB_create(nfeatures=features)
    sift = cv2.SIFT_create(nfeatures=features)
    get_feature_nums = int(features_per * features)

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    kps, dess = sift.detectAndCompute(image, None)
    kpo, deso = orb.detectAndCompute(image, None)
    with open('./models/books_sklearn_knn_sift.pkl', 'rb') as file:
        knn = pickle.load(file)
    with open('./models/books_sklearn_svm_linear_sift.pkl', 'rb') as file:
        svm = pickle.load(file)

    if len(dess) >= get_feature_nums:
        # 假设new_image_features是新图像的特征向量
        new_image_features =dess[:get_feature_nums].reshape(-1).reshape(1, -1)
        # new_image_features = new_image_features.reshape(1, -1)  # 将特征向量转换为一维数组

        prediction = knn.predict(new_image_features)
        print(prediction)

        prediction = svm.predict(new_image_features)
        print(prediction)
    else:
        print(f'{image_path} can\'t get enough sift features from picture ')




if __name__ == '__main__':
    # 读取图像数据集和标签
    images = []
    images = common.get_files_and_folder('./images/train_books', images)
    print("共计 %s 数据" % len(images))
    trainBooksTorchModels(images)
    trainBooksCV2Models(images)
    # trainBooksSklearnModels(images)
    # testBooksSklearnModels('./images/test_books/book_3_0.jpg')

#
# # ## KNN
# model = model.load('./model_books/book_classifier_knn_sift.xml')
# # #
# # # ## SVM
# svm = svm.load('./model_books/book_classifier_svm_sift.xml')
#
# # # 预测新图片
# img0 = 'C:\\Users\\Administrator\\Desktop\\ai_dog\\opencv\\test_books\\book_5_0.jpg'
# img1 = 'C:\\Users\\Administrator\\Desktop\\ai_dog\\opencv\\test_books\\book_7_1.jpg'
# img2 = 'C:\\Users\\Administrator\\Desktop\\ai_dog\\opencv\\test_books\\book_1_2.jpg'
#
#
# ######################KNN
# def getLabelName(img_path, orb, sift, model):
#     test_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
#     # test_img = cv2.imread(img_path)
#     kps, dess = sift.detectAndCompute(test_img, None)
#     kpo, deso = orb.detectAndCompute(test_img, None)
#     print(f'===={img_path}=====', len(deso), len(deso))
#     np_arra = np.array([dess[:90].reshape(-1)]).astype(np.float32)
#     # print(np_arra, len(np_arra), len(np_arra[0]))
#     _, result, a, conf = model.findNearest(np_arra, 5)
#     print(result, _, conf)
#
#
# getLabelName(img0, orb, sift, model) #yueliang
# getLabelName(img1, orb, sift, model) #bailu
# getLabelName(img2, orb, sift, model) #huobi
# ######################KNN
#
#
# ######################随机森林
#
# # from sklearn.ensemble import RandomForestClassifier
# #
# # # 训练数据路径
# # data_path = 'book_dataset/'
# #
# # # 训练随机森林模型
# # rf = RandomForestClassifier(n_estimators=300, random_state=1)
# # rf.fit(np_images, np_labels)
# #
# # def getLabelName2(img_path):
# #     # 预测新图像并计算准确率
# #     new_img = cv2.imread(img_path)
# #     gray = cv2.resize(cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY), (300, 300))
# #     kp, des = sift.detectAndCompute(gray, None)
# #     pred = rf.predict([des[:90].reshape(-1)])
# #     print(pred)
# #
# # getLabelName2(img0)
# # getLabelName2(img1)
# # getLabelName2(img2)
#
#
# ######################随机森林
#
#
#
# ######################SVM
#
# def getLabelName3(img_path, orb, sift, svm):
#     test_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
#     # test_img = cv2.imread(img_path)
#     kps, dess = sift.detectAndCompute(test_img, None)
#     kpo, deso = orb.detectAndCompute(test_img, None)
#     # print(f'===={img_path}=====', len(deso), len(dess))
#
#     np_arra = np.array([dess[:90].reshape(-1)]).astype(np.float32)
#     # print(np_arra, len(np_arra), len(np_arra[0]))
#     pred_labels = svm.predict(np_arra)
#     print(pred_labels[1][0][0])
#
#
# getLabelName3(img0, orb, sift, svm) #yueliang
# getLabelName3(img1, orb, sift, svm) #bailu
# getLabelName3(img2, orb, sift, svm) #huobi


######################SVM

