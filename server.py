from flask import Flask, request, jsonify
import api
import common
import params

server = Flask(__name__)

@server.route('/classify_book', methods=['POST'])
def classify_book():
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)

    resnet_result = {}
    sklearn_result = {}
    # 在这里添加处理文件的代码，例如进行图像识别或特征提取, 只提取书籍特征部分
    flag, file_path = api.pre_get_book(tmp_path)
    try:
        if flag:
           #sklearn_result = api.predict_sklearn_books_models(file_path)
           resnet_result = api.predict_resnet_book(file_path)
    except Exception as e:
        print(e)
    # 返回处理结果给客户端
    response = {'sklearn_result': sklearn_result, 'resnet_result': resnet_result}
    print(response)
    return jsonify(response)


@server.route('/classify_face', methods=['POST'])
def classify_face():
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    # 在这里添加处理文件的代码，例如进行图像识别或特征提取
    face_path = api.pre_get_face(tmp_path)
    if face_path:
        # message, content = api.predict_face_recongnition(face_path)
        message, content = api.predict_resnet_face(face_path)
        if message:
            return jsonify(api.get_face_respone(message, content))
        else:
            return jsonify(api.get_face_respone(message, content))
    else:
        return jsonify(api.get_face_respone(None, "无法获取图片中的人脸数据..."))


@server.route('/borrow_book', methods=['POST'])
def borrow_book():
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    # 处理接收到的书名 - 获取其ID - 如果是未识别的图书， id返回-1
    book_name = request.form.get('book_name')
    # 产生借还书的JSON
    message = api.get_borrow_return_book_message(tmp_path, 'borrow', book_name)
    if message:
        flag, content = common.send_email(params.book_message_title_email, message, params.book_message_file_name)
        return jsonify(api.get_book_respone(message, content))
    else:
        return jsonify(api.get_book_respone(message, "生成借书JSON为空, 请排查server日志"))


@server.route('/return_book', methods=['POST'])
def return_book():
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    # 处理接收到的书名 - 获取其ID - 如果是未识别的图书， id返回-1
    book_name = request.form.get('book_name')
    # 产生借还书的JSON
    message = api.get_borrow_return_book_message(tmp_path, 'return', book_name)
    if message:
        flag, content = common.send_email(params.book_message_title_email, message, params.book_message_file_name)
        return jsonify(api.get_book_respone(message, content))
    else:
        return jsonify(api.get_book_respone(message, "生成借书JSON为空, 请排查server日志"))


@server.route('/add_book', methods=['POST'])
def add_book():
    isbn = request.form.get('isbn')
    # 接收客户端发送的文件
    tmp_path = common.save_requests_file(request)
    message = api.get_add_book_message(tmp_path, isbn)
    if message:
        flag, content = common.send_email(params.book_message_title_email, message, params.book_message_file_name)
        return jsonify(api.get_book_respone(message, content))
    else:
        return jsonify(api.get_book_respone(message, "生成借书JSON为空, 请排查server日志"))



if __name__ == '__main__':
    server.run(host='0.0.0.0',  port=9898)

