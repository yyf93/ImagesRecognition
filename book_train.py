# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import image_extract_features
from book import *
import image_enhance

'''
用各个模型训练图书的分类
'''
def train_books(is_full_train=False):
    """
    使用resnet152去训练图书
    """
    # 获取图书的配置信息
    book_tool = BookTool()

    # 训练集路径
    train_dir = book_tool.book_train_dir

    # 对训练路径的图书进行图像增强
    image_enhance.generate_book_images(train_dir)

    #是否全量训练
    if is_full_train:
        # 全量训练
        # 读取图像数据集
        images = common.get_trains_sub_folder_files(train_dir, [])
        print("全量训练 共计 %s 数据" % len(images))
        train_books_resnet_model(book_tool, images, is_full_train)
    else:
        #增量训练
        # 判断是否有新增的图书图像
        images = book_tool.get_increment_train_images()
        print("增量训练 共计 %s 数据" % len(images))
        # 看是否需要增量训练模型
        if len(images) > 0:
            train_books_resnet_model(book_tool, images, is_full_train)
        else:
            print('未有新增图书素材， 不进行训练')


def train_books_resnet_model(book_tool, images, is_full_train):
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
        if common.is_exit_file(params.model_torch_books_resnet):
            info_dict = torch.load(params.model_torch_books_resnet)
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
        book_isbn = image_path.split('/')[-2]
        uniq_id = book_tool.get_uniq_id(book_isbn)
        print(f'----- path: {image_path} ------- label: {book_isbn} ------ uniq: {uniq_id}')
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
    num_classes = len(book_tool.all_book_infos)
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
    torch.save(state_dict, params.model_torch_books_resnet)

    # 跟新配置文件中该图书已经训练过了
    book_tool.upadte_is_train_to_config(train_labels)


"""
暂时废弃sklearn， 准确率没有resnet高
"""
def train_books_sklearn_models(image_paths, train_all=True):
    '''
    训练sklearn分类模型并评估
    '''
    new_features = []
    new_labels = []
    dirs = []

    for oneImage in image_paths:
        oneImage = oneImage.replace('\\', '/')
        dir = oneImage.split('/')[-2]
        dirs.append(dir)
        image_features = image_extract_features.extract_opencv_features(oneImage)
        if len(image_features) > 0:
            print(f'process ===={oneImage}=====')
            new_features.append(image_features)
            label_read = common.reverseDict(params.book_label_mapping)[dir]
            new_labels.append(label_read)
        else:
            print(f'Invalid image {oneImage}, skipping...')
            continue

    new_features = np.array(new_features).astype(np.float32)
    np_labels = np.array(new_labels).astype(np.int32)
    # 切割数据集
    X_train, X_test, y_train, y_test = train_test_split(new_features, np_labels, test_size=0.2, random_state=42)
    if train_all:
        X_train = new_features
        y_train = np_labels

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
    train_books(is_full_train=True)
    # train_books_sklearn_models(images)
    # testBooksSklearnModels('./images/test_books/dazhanjiqixuexi_0.jpg')
    # testBooksSklearnModels('./images/test_books/manhuasuanfa_0.jpg')
    # testBooksSklearnModels('./images/test_books/huobiyuweilai_0.jpg')

