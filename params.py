# _BASE_URL = 'http://101.34.206.191:9898/'
_BASE_URL = 'http://127.0.0.1:9898/'

chinese_english_map = {'huobiyuweilai': '货币未来', 'manhuasuanfa': '漫画算法', 'bailuyuan': '白鹿原', 'dazhanjiqixuexi': 'Python大战机器学习',
                       'lj': '陆老板', 'lt': '李桃', 'hjs': '黄季盛', 'jbc': '贾博淳', 'yyf': '于一飞', 'Unknown': '不认识'}

book_label_mapping = {'0': 'huobiyuweilai', '1': 'manhuasuanfa', '2': 'bailuyuan', '3': 'dazhanjiqixuexi'}
face_label_mapping = {'0': 'lj', '1': 'lt', '2': 'hjs', '3': 'jbc', '4': 'yyf'}

model_haar_face_cascade = './models/haarcascade_frontalface_default.xml'

features_cv2_sift_nums = 100
features_cv2_sift_percent = 0.9

model_cv2_faces_lbphface = './models/face_cv2_lbphface_sift.xml'
model_cv2_books_knn = './models/books_sklearn_knn_sift_hist.pkl'
model_cv2_books_svm = './models/books_sklearn_svm_linear_sift_hist.pkl'
model_cv2_books_decision_tree = './models/books_sklearn_decision_tree_sift_hist.pkl'
model_cv2_books_random_forest = './models/books_sklearn_random_forest_sift_hist.pkl'
model_torch_books_resnet = './models/books_torch_resnet152_rgb.pth'

params_face_recognition_faces_encoding = './models/face_face_recognition_faces_encodings.npy'
params_face_recognition_faces_names = './models/face_face_recognition_faces_names.npy'





