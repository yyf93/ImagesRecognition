from utils import image_extract_features
from utils.book import *
from config import params
import numpy as np
import pickle


def pre_get_book(tmp_path):
    '''
    预处理
    '''
    return tmp_path



def predict_resnet_book(unknown_image_path):
    '''
    RESNET
    '''
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torch.nn.functional as F
    try:
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
        #isbn = common.reverse_dict(book_tool.current_book_isbn_id_json)[str(predicted_label)]
        name = common.reverse_dict(book_tool.book_name_id)[str(predicted_label)]
        print(f'predicted_label: {predicted_label} ------  isbn_uniq_id_dict: {book_tool.current_book_isbn_id_json}')
        return json.dumps({"resnet_result": {name: confidence}}, ensure_ascii=False), ""
    except Exception as e:
        return None, str(e)



def get_book_respone(json_str, content=""):
    """
    将server book信息返回给client
    """
    if json_str is None:
        json_str = "{}"
    message_json = json.loads(json_str)
    if "imgBytes" in message_json.keys():
        message_json.pop("imgBytes")
    #print(message_json)
    if content == "":
        message_json["is_success"] = "True"
        message_json["error_message"] = content
    else:
        message_json["is_success"] = "False"
        message_json["error_message"] = content
    return message_json



def get_borrow_return_book_message(img_path, book_type, book_name):
    """
    借还书JSON生成
    """
    book_tool = BookTool()
    return book_tool.get_borrow_return_json_file(img_path, book_type, book_name)


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




if __name__ == '__main__':
    #print(predict_face_recongnition('./images/test_faces/lj_01.jpg'))
    #img_file = './images/test_books/barcode4.jpg'
    #img_file = './images/train_books/9787506394314/9787506394314_angle60.jpg'
    #print(predict_resnet_book(img_file))
    #print(get_add_book_message(img_file, None))
    #face_file = './images/test_faces/pb_0.jpg'
    #pre_get_face(face_file)
    #print(predict_resnet_face(face_file))
    #print(predict_resnet_face(face_file))
    #send_email(img_file='./tmp/barcode.jpg', action='borrow', book_id='111', book_name='test')

    # 循环发送书籍信息到server
    import os
    exis_isbn = ['9787020106684', '9787020145980', '9787115409584', '9787115537157', '9787508694740',
                 '9787201102498', '9787208061644', '9787208115132', '9787308190138', '9787506394314',
                 '9787530218242', '9787536090002', '9787521603774', '9787530217481', '9787530218242',
                 '9787536090002', '9787536484276', '9787536692930', '9787542673053', '9787544291163',
                 '9787550238763', '9787554615430', '9787561351284', '9787565423031', '9787207059055',
                 '9787532781751', '9787544298995', '9787530217948'
                 ]
    for root, dirs, filenames in os.walk('data/images/train_books'):
        for dir in dirs:
            if dir in exis_isbn:
                continue
            json_str = get_add_book_message(f'./images/train_books/{dir}/{dir}.jpg', dir)
            print(json_str)
            common.send_email(params.book_message_title_email, json_str, params.book_message_file_name)
            time.sleep(5)
            print(f'process ./images/train_books/{dir}/{dir}.jpg DONE!!!')

    # json_str = get_add_book_message(f'./images/train_books/9787508684031/9787508684031.jpg', '9787508684031')
    # common.send_email(params.book_message_title_email, json_str, params.book_message_file_name)
