import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import params
import common
import torch
import torch.nn as nn
import torchvision.models as models
import image_extract_features

'''
用各个模型训练图书的分类
'''

def train_books_resnet_model(images):
    '''
    加载预训练的ResNet152层模型
    '''
    resnet = models.resnet152(pretrained=True)
    resnet.fc = nn.Identity()  # 去除最后的全连接层

    # 提取训练集中的特征
    train_features = []
    train_labels = []
    for image_path in images:
        print(f'----- {image_path}')
        image_path = image_path.replace('\\', '/')
        dir = image_path.split('/')[-2]
        train_labels.append(int(common.reverseDict(params.book_label_mapping)[dir]))
        features = image_extract_features.extract_resnet_features(image_path, resnet)
        train_features.append(features)

    np_features = torch.cat(train_features, dim=0)
    np_labels = torch.tensor(train_labels)
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(np_features, np_labels, test_size=0.2, random_state=42)

    # num_classes为类别的数量
    num_classes = len(params.book_label_mapping)
    # 构建分类器模型
    classifier = nn.Linear(np_features.shape[1], num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # 训练分类器
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Models_Evaluation.txt
    with torch.no_grad():
        outputs = classifier(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Validation Accuracy: {accuracy}")
    # 保存模型
    torch.save(classifier.state_dict(), params.model_books_torch_resnet)


def train_books_sklearn_models(image_paths):
    '''
    训练sklearn分类模型并评估
    '''
    new_sift_features = []
    new_labels = []
    dirs = []

    for oneImage in image_paths:
        oneImage = oneImage.replace('\\', '/')
        dir = oneImage.split('/')[-2]
        dirs.append(dir)
        image_features = image_extract_features.extract_opencv_features(oneImage)
        if len(image_features) > 0:
            print(f'process ===={oneImage}=====')
            new_sift_features.append(image_features)
            label_read = common.reverseDict(params.book_label_mapping)[dir]
            new_labels.append(label_read)
        else:
            print(f'Invalid image {oneImage}, skipping...')
            continue

    np_sift_images = np.array(new_sift_features).astype(np.float32)
    np_labels = np.array(new_labels).astype(np.int32)
    # 切割数据集
    X_train, X_test, y_train, y_test = train_test_split(np_sift_images, np_labels, test_size=0.2, random_state=42)

    # 训练分类器
    knn = KNeighborsClassifier(n_neighbors=len(set(np_labels)), metric='euclidean')
    knn.fit(X_train, y_train)

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    random_forest = RandomForestClassifier(n_estimators=200)
    random_forest.fit(X_train, y_train)

    # 模型评估
    knn_y_pred = knn.predict(X_test)
    print('----- KNN -----')
    print(classification_report(y_test, knn_y_pred))

    svm_y_pred = svm.predict(X_test)
    print('----- SVM -----')
    print(classification_report(y_test, svm_y_pred))

    decision_tree_y_pred = decision_tree.predict(X_test)
    print('----- DecisionTree -----')
    print(classification_report(y_test, decision_tree_y_pred))

    random_forest_y_pred = svm.predict(X_test)
    print('----- RandomForest -----')
    print(classification_report(y_test, random_forest_y_pred))

    # 模型保存
    with open(params.model_cv2_books_knn, 'wb') as file:
        pickle.dump(knn, file)
    with open(params.model_cv2_books_svm, 'wb') as file:
        pickle.dump(svm, file)
    with open(params.model_cv2_books_decision_tree, 'wb') as file:
        pickle.dump(decision_tree, file)
    with open(params.model_cv2_books_random_forest, 'wb') as file:
        pickle.dump(random_forest, file)


def testBooksSklearnModels(image_path):
    '''
    验证集测试模型
    '''
    new_image_features = image_extract_features.extract_opencv_features(image_path)
    with open('./models/books_sklearn_knn_sift_hist.pkl', 'rb') as file:
        knn = pickle.load(file)
    with open('./models/books_sklearn_svm_linear_sift_hist.pkl', 'rb') as file:
        svm = pickle.load(file)
    if len(new_image_features) > 0:
        prediction = knn.predict([new_image_features])
        print(prediction)

        prediction = svm.predict([new_image_features])
        print(prediction)
    else:
        print(f'{image_path} can\'t get enough sift features from picture ')


if __name__ == '__main__':
    # 读取图像数据集和标签
    images = []
    images = common.get_files_and_folder('./images/train_books', images)
    print("共计 %s 数据" % len(images))
    # train_books_resnet_model(images)
    train_books_sklearn_models(images)
    # testBooksSklearnModels('./images/test_books/dazhanjiqixuexi_0.jpg')
    # testBooksSklearnModels('./images/test_books/manhuasuanfa_0.jpg')
    # testBooksSklearnModels('./images/test_books/huobiyuweilai_0.jpg')

