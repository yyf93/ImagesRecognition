import os
import datetime
import params


def getChineseName(engName):
    """
    英文名转中文名
    """
    return params.chinese_english_map[engName]


def getCurrentTimeStr():
    """
    获取当前时间
    """
    now = datetime.datetime.now()
    timestamp = now.timestamp()
    time_format = '%Y%m%d%H%M%S'
    return datetime.datetime.fromtimestamp(timestamp).strftime(time_format)


def get_files_and_folder(path, images):
    """
    获取一个路径下的所有文件完整路径和路径的最后一个文件夹的名称
    :param path: str, 路径名
    :return: list, str， 该路径下的所有文件的完整路径; str, 路径的最后一个文件夹的名称
    """
    # 判断路径是否存在
    if not os.path.exists(path):
        print(f"Error: The path {path} does not exist.")
        return [], ''
    for root, dirs, filenames in os.walk(path):
        print("--------", root)
        for dir in dirs:
            files_and_folders = os.listdir(root+'/'+dir)
            # 过滤出当前工作目录下的所有文件
            files_only = [f for f in files_and_folders if os.path.isfile(os.path.join(root+'/'+dir, f))]
            for filename in files_only:
                full_path = root + '/' + dir + '/' + filename
                # print('---------------------', full_path)
                images.append(full_path)
    return images


def reverseDict(dict):
    """
    反转字典
    """
    new_dict = {}
    for k, v in dict.items():
        new_dict[v] = k
    return new_dict


