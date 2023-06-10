from flask import Flask, request, jsonify
import api
import common

app = Flask(__name__)

@app.route('/classify_book', methods=['POST'])
def classify_book():
    # 接收客户端发送的文件
    time_str = common.getCurrentTimeStr()
    tmp_path = f'./tmp/book_tmp_{time_str}.jpg'
    file = request.files['image']
    file.save(tmp_path)
    knn_result = ''
    svm_result = ''
    resnet_result = ''
    knn_confidence = ''
    svm_confidence = ''
    resnet_confidence = ''
    # 在这里添加处理文件的代码，例如进行图像识别或特征提取, 只提取书籍特征部分
    flag, file_path = api.pre_get_book(tmp_path)
    if flag:
       sklearn_result = api.predict_sklearn_books_models(file_path)
       resnet_result = api.predict_resnet(file_path)
    # 返回处理结果给客户端
    response = {'sklearn_result': sklearn_result, 'resnet_result': resnet_result}
    print(response)
    return jsonify(response)


@app.route('/classify_people', methods=['POST'])
def classify_people():
    # 接收客户端发送的文件
    time_str = common.getCurrentTimeStr()
    tmp_path = f'./tmp/people_tmp_{time_str}.jpg'
    file = request.files['image']
    file.save(tmp_path)
    lbphface_result = ''
    face_recognition_label_result = ''
    # 在这里添加处理文件的代码，例如进行图像识别或特征提取
    flag, face_path = api.pre_get_face(tmp_path)
    if flag:
        lbphface_result = api.predict_cv2_lbphface(face_path)
        face_recognition_label_result = api.predict_face_recongnition(face_path)
    # 返回处理结果给客户端
    response = {'lbphface_result': lbphface_result,
                'face_recognition_label_result': face_recognition_label_result}
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=9898)

