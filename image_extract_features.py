import cv2
import params


def extract_face_haarcascade_features(img_path):
    '''
    用haarcascade_frontalface_default.xml去图中截取人脸
    '''
    faces_features = []
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
    - Image.open()打开图片,convert('RGB')转换为RGB模式
    - transforms对图片进行预处理,包括`Resize`、`CenterCrop`、`ToTensor`和`Normalize`
    - input_tensor.unsqueeze(0)添加维度,作为模型输入
    - with torch.no_grad():关闭梯度计算,进行预测
    - features = model(input_batch)使用模型对图片进行特征提取
    - return features返回提取到的特征
    这个函数的总体作用是:加载图片,对图片进行预处理,输入模型进行特征提取,并返回提取到的特征。
    '''
    import torchvision.transforms as transforms
    from PIL import Image
    import torch
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        #这是图像预处理的transforms,将图像大小重新调整为256x256
        transforms.Resize(256),
        #图像中心裁剪出224x224的区域
        transforms.CenterCrop(224),
        #图像转换为Tensor,并调整其范围到[0, 1]。
        transforms.ToTensor(),
        #归一化 有利于模型训练- mean=[0.485, 0.456, 0.406]:三个通道的均值。std=[0.229, 0.224, 0.225]:三个通道的标准差
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
