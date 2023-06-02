import os

chinese_english_map = {'huobiyuweilai': '货币未来', 'manhuasuanfa': '漫画算法', 'bailuyuan': '白鹿原', 'dazhanjiqixuexi': 'Python大战机器学习',
                       'lt': '李桃', 'hjs': '黄季盛', 'jbc': '贾博淳', 'yyf': '于一飞', 'Unknown': '不认识'}

book_label_mapping = {'0': 'huobiyuweilai', '1': 'manhuasuanfa', '2': 'bailuyuan', '3': 'dazhanjiqixuexi'}
face_label_mapping = { '0': 'lt', '1': 'hjs', '2': 'jbc', '3': 'yyf'}


model_haar_face_cascade = './models/haarcascade_frontalface_default.xml'

model_faces_cv2_lbphface = './models/face_cv2_lbphface_sift.xml'
model_books_cv2_knn = './models/books_cv2_knn_sift.xml'
model_books_cv2_svm = './models/books_cv2_svm_linear_sift.xml'
model_books_torch_resnet = './models/books_torch_resnet50_rgb.pth'

params_face_recognition_faces_encoding = './models/face_face_recognition_faces_encodings.npy'
params_face_recognition_faces_names = './models/face_face_recognition_faces_names.npy'





