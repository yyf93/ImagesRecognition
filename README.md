# ImagesRecognition

## 环境要求:
pip install flask opencv-python opencv-contrib-python torch face_recognition scikit-learn

## 项目简介:
人脸和图书的图像识别并分类.

## 如何使用

### 数据预处理

#### ```python image_enhance.py``` 
使数据增强书
籍数据我们增强方式旋转+水平翻转+双边滤波去噪
人脸数据我们增强方式水平翻转+中值滤波去噪

### 训练模型

#### ```python books_train.py```
#### ```python faces_train.py```
书籍识别用了sklearn的很多模型和resnet模型, 评估后, 准确率最高的是resnet152模型
人脸识别用了cv2的lbph_face模型和face_recognition模型，评估后, 准确率最高的是face_recognition模型

### 模型评估
先用测试集和训练集去测试模型的准确率
再用实际的图片去找到最优模型。

### 打开服务
#### ```python server.py```
主要服务就是识别人脸和图书

### 调用服务
#### ```python client.py```
发送请求去获取服务端的分类结果

