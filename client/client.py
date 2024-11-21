import requests
from client_api import book_api, face_api
from utils import camera
from config import params


def classify_book_image(file_path=None):
    '''
    client识别书籍图像
    '''
    # classify_books
    url = f'{params._BASE_URL}classify_book'  # 服务端的URL
    book = ['Unknown']
    # 获取图像流
    files = camera.get_bytes_from_capture_or_file(file_path, 'classify_book')
    # 发送请求
    response = requests.post(url, files=files)
    # 解析响应数据
    result = response.json()
    if result is not None:
        book = book_api.get_book(result)
    return book


def classify_people_image(file_path=None):
    '''
    client识别人脸图像
    '''
    # classify_face
    url = f'{params._BASE_URL}classify_face'
    #print(url)
    face = ['Unknown']
    # 获取图像流
    files = camera.get_bytes_from_capture_or_file(file_path, 'classify_face')
    # 发送请求
    response = requests.post(url, files=files)
    # 解析响应数据
    result = response.json()
    if result is not None:
        face = face_api.get_face(result)
    return face


def flow_add_book(file_path=None, isbn=None):
    '''
    图书流程-新增书, 流程做到一半TODO!
    '''
    url = f'{params._BASE_URL}add_book'  # 服务端的URL
    # 获取图像流
    files = camera.get_bytes_from_capture_or_file(file_path, 'add_book')
    data = {'isbn': isbn}
    # 发送请求
    response = requests.post(url, files=files, data=data)
    # 解析响应数据
    result = response.json()
    return result


def flow_borrow_return_book(book_type, book_name, file_path=None):
    '''
    图书流程-借还书
    '''
    # borrow/return_books
    url = f'{params._BASE_URL}{book_type}_book'  # 服务端的URL

    # 获取图像流
    files = camera.get_bytes_from_capture_or_file(file_path, f'{book_type}_book')
    # 发送请求
    data = {'book_name': book_name}
    response = requests.post(url, files=files, data=data)

    # 解析响应数据
    result = response.json()
    return result


if __name__ == '__main__':
    # file_path = sys.argv[1]
    book_path = './images/test_books/ss.png'
    #print(classify_book_image(book_path))
    people_path = './images/test_faces/yyf_0.jpg'
    #print(classify_people_image(people_path))
    img_test = './images/test_books/barcode.jpg'
    add_img = './images/test_books/barcode3.jpg'
    # respone = flow_borrow_return_book('borrow', '货币未来', file_path=img_test)
    # print(respone)
    # print(type(respone["is_success"]))
    # print(respone["is_success"])
    print(flow_add_book(book_path, '9780008227876'))
    # print(flow_add_book(add_img))
    face_img = './images/test_faces/pb_0.jpg'
    #print(classify_book_image(book_path))
    #print(classify_people_image(face_img))
