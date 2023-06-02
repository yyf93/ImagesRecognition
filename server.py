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
       knn_result, knn_confidence = api.knn(file_path)
       svm_result, svm_confidence = api.svm(file_path)
       resnet_result, resnet_confidence = api.resnetBooks(file_path)
    # 返回处理结果给客户端
    response = {'knn_result': knn_result, 'svm_result': svm_result, 'resnet_result': resnet_result,
                'resnet_confidence': str(resnet_confidence), 'svm_confidence': str(svm_confidence),
                'knn_confidence': str(knn_confidence)}
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
    lbphface_confidence = ''
    face_recognition_label_result = ''
    face_recognition_confidence = ''
    # 在这里添加处理文件的代码，例如进行图像识别或特征提取
    flag, face_path = api.pre_get_face(tmp_path)
    if flag:
        lbphface_result, lbphface_confidence = api.lbphface(face_path)
        face_recognition_label_result, face_recognition_confidence = api.faceRecongnition(face_path)
    # 返回处理结果给客户端
    response = {'lbphface_result': lbphface_result, 'lbphface_confidence': str(lbphface_confidence),
                'face_recognition_label_result': face_recognition_label_result,
                'face_recognition_confidence': str(face_recognition_confidence)}
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=9898)

