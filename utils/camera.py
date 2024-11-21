# -*- coding: utf-8 -*-
from utils import common
from config import params
import camera
import shutil
import cv2
import time

class CameraTool:
    '''
    调用摄像头接口
    '''

    @staticmethod
    def get_bytes_from_capture_or_file(file_path, file_prefix):
        """
        从摄像头或者文件中获取文件字节
        """
        time_str = common.get_current_time_str()
        tmp_path = f'{params.base_directory}/tmp/{file_prefix}_{time_str}.jpg'.replace('\\', '/')
        print(f'client local images: {file_path}  send images : {tmp_path}')
        if file_path is None:
            camera.capturePhoto(file_name=tmp_path)
        else:
            shutil.copyfile(file_path, tmp_path)
        img_file = open(tmp_path, 'rb')
        return {'image': img_file}

    @staticmethod
    def capturePhoto(file_name=None):
        # 初始化摄像头
        cap = cv2.VideoCapture(0)

        # 检查是否打开摄像头成功
        if not cap.isOpened():
            print("Unable to open camera")

        # # 设置视频参数
        #cap.set(3, 720)   # 设置图像宽度
        #cap.set(4, 720)   # 设置图像高度

        # 读取每一帧并判断是否读取成功
        ret, frame = cap.read()
        if file_name is None:
            file_name = './tmp/camera_%s.jpg' % time.time()
        if ret == True:
            # # 按下s键拍照并保存
            # if cv2.waitKey(1) & 0xFF == ord('s'):
            # 将当前帧保存为照片
            cv2.imwrite(file_name, frame)

        # 释放资源并关闭窗口
        cap.release()
        cv2.destroyAllWindows()
        return file_name

