import cv2
import common

'''
数据增强 -
1. 对图像进行旋转
2. 对图像进行缩放 （不用）
3. 对图像进行翻转
4. 对图像进行去噪音

'''

def resize_image(img_path, small_per_list = [0.8, 0.6, 0.4, 0.2]):
    '''
    图像缩放
    '''
    pre_name = "".join(img_path.split('.')[:-2])
    # 按比例缩小图片尺寸
    from PIL import Image
    im = Image.open(img_path)
    for resize in small_per_list:
        (x, y) = im.size  # 读取图片尺寸（像素）
        x_s = int(x * resize)  # 定义缩小后的标准宽度
        y_s = int(y * resize)   # 基于标准宽度计算缩小后的高度
        out = im.resize((x_s, y_s), Image.LANCZOS)  # 改变尺寸，保持图片高品质
        out.save(f'{pre_name}_{resize}.jpg')


def rotated_image(img_path, interval=30):
    '''
    图像旋转
    '''
    files = []
    pre_name = ".".join(img_path.split('.')[:-1])
    img = cv2.imread(img_path)

    for angle in range(0, 360, interval):
        # 计算旋转中心和旋转矩阵
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 进行旋转变换
        rotated_img = cv2.warpAffine(img, M, (width, height))
        # 保存旋转后的图片
        new_img_file_path = f'{pre_name}_angle{angle}.jpg'
        files.append(new_img_file_path)
        cv2.imwrite(new_img_file_path, rotated_img)
        print(f'save {new_img_file_path}')
    return files


def generate_book_images(img_path):
    '''
    主函数 - 图书增强
    注意： 初始化中的文件夹中的子文件夹的照片最好唯一，不然会生成大量的照片！！！
    '''
    # 获取路径下面的子文件夹中的所有文件
    for files in common.get_files_and_folder(img_path, []):
        files = files.replace('\\', '/')
        img_file_whole_path = files
        for i in rotated_image(img_file_whole_path):
            for j in flip_images(i):
                remove_noise_images(j, ['bilateral'])


def generate_face_images(img_path):
    '''
    主函数 - 人脸增强
    '''
    # 获取路径下面的子文件夹中的所有文件
    for files in common.get_files_and_folder(img_path, []):
        files = files.replace('\\', '/')
        img_file_whole_path = files
        for j in flip_images(img_file_whole_path):
            remove_noise_images(j, ['median'])


def flip_images(img_path):
    '''
    图像翻转
    '''
    files = []
    pre_name = ".".join(img_path.split('.')[:-1])
    # 读取原图像
    img = cv2.imread(img_path)
    # 水平翻转
    flip_horizontal = cv2.flip(img, 1)
    # 垂直翻转
    # flip_vertical = cv2.flip(img, 0)
    # 水平垂直翻转
    # flip_both = cv2.flip(img, -1)
    filename = f'{pre_name}_flip_horizontal.jpg'
    cv2.imwrite(filename, flip_horizontal)
    files.append(filename)
    print(f'save {filename}')
    return files


def remove_noise_images(img_path, noises):
    '''
    图像去噪音
    '''
    files = []
    pre_name = ".".join(img_path.split('.')[:-1])
    # 读取原图像
    img = cv2.imread(img_path)
    for noise in noises:
        if noise == 'gaussian':
            gaussian_filename = f'{pre_name}_gaussian.jpg'
            # 高斯滤波去噪
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(gaussian_filename, gaussian)
            files.append(gaussian_filename)
            print(f'save {gaussian_filename}')
        elif noise == 'median':
            median_filename = f'{pre_name}_medianBlur.jpg'
            # 中值滤波去噪
            median = cv2.medianBlur(img, 5)
            cv2.imwrite(median_filename, median)
            files.append(median_filename)
            print(f'save {median_filename}')
        elif noise == 'bilateral':
            bilateral_filename = f'{pre_name}_bilateral.jpg'
            # 双边滤波去噪
            bilateral = cv2.bilateralFilter(img, 9, 75, 75)
            cv2.imwrite(bilateral_filename, bilateral)
            files.append(bilateral_filename)
            print(f'save {bilateral_filename}')
    return files


if __name__ == '__main__':
    path = './images'
    # remove_noise_images('C:\\Users\\Administrator\\Desktop\\ImagesRecognition\\images\\train_books\\bailuyuan\\bailuyuan_0_1.jpg')
    generate_book_images(f'{path}/train_books')
    generate_face_images(f'{path}/train_faces')


