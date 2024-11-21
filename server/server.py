import sys

from flask import Flask, request, jsonify
from server.api import book_api, face_api
from utils import common
from config import params

server = Flask(__name__)

@server.route('/classify_book', methods=['POST'])
def classify_book():
    print("process classify_book...")
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    # 在这里添加处理文件的代码，例如进行图像识别或特征提取, 只提取书籍特征部分
    file_path = book_api.pre_get_book(tmp_path)
    if file_path:
        # sklearn_result = api.predict_sklearn_books_models(file_path)
        message, content = book_api.predict_resnet_book(file_path)
        if message:
            return jsonify(book_api.get_book_respone(message, content))
        else:
            return jsonify(book_api.get_book_respone(message, content))
    else:
        return jsonify(book_api.get_book_respone(None, "为获取到图中书籍，请排查server日志..."))


@server.route('/classify_face', methods=['POST'])
def classify_face():
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    # 在这里添加处理文件的代码，例如进行图像识别或特征提取, 可能会有多个脸的图像， 都需要处理
    face_paths = face_api.pre_get_face(tmp_path)
    if face_paths:
        res_message = '{}'
        for one_face_path in face_paths:
            # resnet的模型预测没有 face_recognition模型预测准
            # message, content = api.predict_resnet_face(one_face_path)
            message, content = face_api.predict_face_recongnition(one_face_path)
            if message:
                res_message = face_api.merge_face_recongnition_message(message, res_message)
        return jsonify(face_api.get_face_respone(res_message, content))

    else:
        return jsonify(face_api.get_face_respone(None, "无法获取图片中的人脸数据..."))


@server.route('/borrow_book', methods=['POST'])
def borrow_book():
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    # 处理接收到的书名 - 获取其ID - 如果是未识别的图书， id返回-1
    book_name = request.form.get('book_name')
    # 产生借还书的JSON
    message = book_api.get_borrow_return_book_message(tmp_path, 'borrow', book_name)
    if message:
        flag, content = common.send_email(params.book_message_title_email, message, params.book_message_file_name)
        return jsonify(book_api.get_book_respone(message, content))
    else:
        return jsonify(book_api.get_book_respone(message, "生成借书JSON为空, 请排查server日志"))


@server.route('/return_book', methods=['POST'])
def return_book():
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    # 处理接收到的书名 - 获取其ID - 如果是未识别的图书， id返回-1
    book_name = request.form.get('book_name')
    # 产生借还书的JSON
    message = book_api.get_borrow_return_book_message(tmp_path, 'return', book_name)
    if message:
        flag, content = common.send_email(params.book_message_title_email, message, params.book_message_file_name)
        return jsonify(book_api.get_book_respone(message, content))
    else:
        return jsonify(book_api.get_book_respone(message, "生成借书JSON为空, 请排查server日志"))


@server.route('/add_book', methods=['POST'])
def add_book():
    isbn = request.form.get('isbn')
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    message = book_api.get_add_book_message(tmp_path, isbn)
    if message:
        flag, content = common.send_email(params.book_message_title_email, message, params.book_message_file_name)
        return jsonify(book_api.get_book_respone(message, content))
    else:
        return jsonify(book_api.get_book_respone(message, "生成借书JSON为空, 请排查server日志"))



if __name__ == '__main__':
    server.run(host='0.0.0.0',  port=9898)

