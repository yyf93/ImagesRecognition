import requests
import camera
import api
import params

def classify_book_image(file_path=None):
    '''
    client识别书籍图像
    '''
    # classify_books
    url = f'{params._BASE_URL}classify_book'  # 服务端的URL
    #print(url)
    book = ['Unknown']
    if file_path is None:
        file = open(camera.capturePhoto(), 'rb')
    else:
        file = open(file_path, 'rb')
    # 构建请求数据
    files = {'image': file}
    # 发送请求
    response = requests.post(url, files=files)
    # 解析响应数据
    result = response.json()
    print(result)
    if result is not None:
        book = api.get_book(result)
    return book


def classify_people_image(file_path=None):
    '''
    client识别人脸图像
    '''
    # classify_books
    url = f'{params._BASE_URL}classify_people'  # 服务端的URL
    #print(url)
    face = ['Unknown']
    if file_path is None:
        file = open(camera.capturePhoto(), 'rb')
    else:
        file = open(file_path, 'rb')
    # 构建请求数据
    files = {'image': file}
    # 发送请求
    response = requests.post(url, files=files)
    # 解析响应数据
    result = response.json()
    print(result)
    if result is not None:
        face = api.get_face(result)
    return face


if __name__ == '__main__':
    # file_path = sys.argv[1]
    book_path = './images/test_books/manhuasuanfa_0.jpg'
    people_path = './images/test_faces/yyf_0.jpg'
    print(classify_book_image(book_path))
    print(classify_people_image(people_path))
