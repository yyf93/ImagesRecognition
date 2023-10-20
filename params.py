import os

# 基准目录，通常是你的服务器应用程序所在的目录
base_directory = os.path.dirname(__file__)

#_BASE_URL = 'xxxxxxxxxxxxxxx'
_BASE_URL = 'http://127.0.0.1:9898/'

isbn_token = 'xxxxxxxxxxxxxxx'

book_train_dir = f'{base_directory}/images/train_books'
book_config_json = f'{base_directory}/config/books.json'

face_train_dir = f'{base_directory}/images/train_faces'
face_config_json = f'{base_directory}/config/faces.json'

model_haar_face_cascade = f'{base_directory}/models/haarcascade_frontalface_default.xml'

features_cv2_sift_nums = 100
features_cv2_sift_percent = 0.9

model_cv2_faces_lbphface = f'{base_directory}/models/face_cv2_lbphface_sift.xml'
model_cv2_books_knn = f'{base_directory}/models/books_sklearn_knn_sift_hist.pkl'
model_cv2_books_svm = f'{base_directory}/models/books_sklearn_svm_linear_sift_hist.pkl'
model_cv2_books_decision_tree = f'{base_directory}/models/books_sklearn_decision_tree_sift_hist.pkl'
model_cv2_books_random_forest = f'{base_directory}/models/books_sklearn_random_forest_sift_hist.pkl'
model_torch_books_resnet = f'{base_directory}/models/books_torch_resnet152_rgb.pth'
model_torch_faces_resnet= f'{base_directory}/models/faces_torch_resnet152_rgb.pth'

params_face_recognition_faces_encoding = f'{base_directory}/models/face_face_recognition_faces_encodings.npy'
params_face_recognition_faces_names = f'{base_directory}/models/face_face_recognition_faces_names.npy'

scert_aes_key = "xxxxxxxxxxxxxxx"
book_message_file_name = 'xxxxxxxxxxxxxxx.json'
book_message_title_email = 'xxxxxxxxxxxxxxx'
mail_host = "smtp.qq.com"
mail_user = "xxxxxxxxxxxxxxx"
mail_pass = "xxxxxxxxxxxxxxx"
mail_port = 465
mail_sender = 'xxxxxxxxxxxxxxx@qq.com'

mail_receivers = ['xxxxxxxxxxxxxxx@qq.com', 'xxxxxxxxxxxxxxx']

