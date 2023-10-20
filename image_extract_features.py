import cv2
import params
import common

def extract_face_from_img_to_file(img_path):
    '''
    提取图像中的人脸，将数据再写回img中, 只取图中一张脸
    '''
    #extract_face_haarcascade_features这个会提取到非人脸的数据， 但是他运行很快，准确率低
    #facess = extract_face_haarcascade_features(img_path)
    faces = extract_face_face_recognition_features(img_path)
    if len(faces) > 0:
        for ele in faces:
            cv2.imwrite(img_path, ele)
            return True
    else:
        common.delete_file(img_path)
        print(f"该图片 {img_path} - 无法提取人脸")
        return False


def extract_face_face_recognition_features(img_path, is_color=True):
    import face_recognition
    '''
    用haarcascade_frontalface_default.xml去图中截取人脸
    '''
    faces_features = []
    # 加载图像并识别人脸
    face = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(face)

    # 使用opencv加载原始图像
    if is_color:
        image = cv2.imread(img_path)
    else:
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

    # 遍历每个识别到的人脸，并保存为新的图片
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # 提取人脸
        face_image = image[top:bottom, left:right]
        faces_features.append(face_image)
    return faces_features


def extract_face_haarcascade_features(img_path, is_color=True):
    '''
    用haarcascade_frontalface_default.xml去图中截取人脸
    '''
    faces_features = []
    # 读取灰色的话 cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    if is_color:
        image = cv2.imread(img_path)
    else:
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # Haar特征文件路径
    haar_face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    faces = haar_face_cascade.detectMultiScale(image, 1.3, 5)
    len_faces = len(faces)
    print(f'{img_path} find {len_faces} faces!!')
    for (x, y, w, h) in faces:
        faces_features.append(image[y:y + h, x:x + w])
    return faces_features


def extract_resnet_features(image_path, model):
    '''
    加载书籍图像并进行特征提取
    '''
    import torchvision.transforms as transforms
    from PIL import Image
    import torch
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(input_batch)
    return features


def extract_color_histogram(image):
    '''
    Opencv获取颜色直方图进行特征提取
    '''
    from sklearn.preprocessing import normalize
    # 将图像转换为HSV颜色空间
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 计算颜色直方图
    hist = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    # 归一化直方图
    hist = normalize([hist.flatten()])
    return hist


def merge_features_sift_hist(sift_features, hist_features):
    '''
    合并特征颜色直方图+SIFT浅层特征
    '''
    features = []
    for i in sift_features[0]:
        features.append(i)
    for i in hist_features[0]:
        features.append(i)
    return features


def extract_opencv_features(image_path):
    '''
    加载图像并用opencv进行特征提取， sift+hist
    '''
    from sklearn.preprocessing import normalize
    features = params.features_cv2_sift_nums
    features_per = params.features_cv2_sift_percent
    get_feature_nums = int(features_per * features)
    sift = cv2.SIFT_create(nfeatures=features)
    org_image = cv2.imread(image_path)

    hist_features = extract_color_histogram(org_image)
    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
    kps, dess = sift.detectAndCompute(image, None)
    image_features = []
    if len(dess) >= get_feature_nums:
        sift_features = normalize([dess[:get_feature_nums].reshape(-1)])
        image_features = merge_features_sift_hist(sift_features, hist_features)
    else:
        print(f'{image_path} can\'t get enough sift features from picture ')
    return image_features
