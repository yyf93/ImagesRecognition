import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import normalize
import cv2


def extract_face_haarcascade_features(img_path):
    '''
    用haarcascade_frontalface_default.xml去图中截取人脸
    '''
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    face = None
    # Haar特征文件路径
    haar_face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    faces = haar_face_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
    return face


def extract_resnet_features(image_path, model):
    '''
    加载书籍图像并进行特征提取
    '''
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


def extract_opencv_features(image_path, features, features_per):
    '''
    加载图像并用opencv进行特征提取， sift+hist
    '''
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
