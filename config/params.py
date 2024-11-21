import os

# 基准目录，通常是你的服务器应用程序所在的目录
base_directory = os.path.dirname(__file__)

#_BASE_URL = 'http://101.34.206.191:9898/'
_BASE_URL = 'http://127.0.0.1:9898/'

isbn_token = 'ity6sVCqbxhIaQkM98qovyDOjenBgvPu'

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

scert_aes_key = "doggandanerniubi"
book_message_file_name = 'ai_dog_messages.json'
book_message_title_email = 'AI Dog Book Messages'
mail_host = "smtp.qq.com"
mail_user = "553592045"
mail_pass = "pzrkrqpdlxtobaic"
mail_port = 465
mail_sender = '553592045@qq.com'
#, 'taotao.dai@citi.com'
mail_receivers = ['553592045@qq.com', 'taotao.dai@citi.com']

book_location="花木17F"
baidu_api_fanyi_appid = "20231122001888830"
baidu_api_fanyi_salt = "553592045"
baidu_api_fanyi_scret_key = "ZFj8GtBBjn5rWiI8PlF0"

