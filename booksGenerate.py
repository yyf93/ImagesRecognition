import cv2
import numpy as np
import common

# GENERATE IMG
def smallImage(img_path):
    index = img_path.split('_')[-2]
    # 按比例缩小图片尺寸
    from PIL import Image
    im = Image.open(img_path)
    small_per_list = [0.6, 0.3]
    for smallper in small_per_list:
        (x, y) = im.size  # 读取图片尺寸（像素）
        x_s = int(x * smallper)  # 定义缩小后的标准宽度
        y_s = int(y * smallper)   # 基于标准宽度计算缩小后的高度
        out = im.resize((x_s, y_s), Image.LANCZOS)  # 改变尺寸，保持图片高品质
        out.save(f'{path}\\{dir}_{index}_{smallper}.jpg')


def generateImages(img_path):
    for files in common.get_files_and_folder(img_path, []):
        files = files.replace('\\', '/')
        img_file_whole_path = files
        dir = files.split('/')[-2]
        # smallImage(img_path)
        img = cv2.imread(img_file_whole_path)

        for angle in range(10, 360, 30):
            # 计算旋转中心和旋转矩阵
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # 进行旋转变换
            rotated_img = cv2.warpAffine(img, M, (width, height))
            # 保存旋转后的图片
            new_img_file_path = f'{img_path}/{dir}/{dir}_{angle}_1.jpg'
            cv2.imwrite(new_img_file_path, rotated_img)
            print(f'generate {dir} more {new_img_file_path}')
            # smallImage(new_img_file_path)




if __name__ == '__main__':
    path = './images'
    img_books_path = f'{path}/train_books'
    generateImages(img_books_path)


