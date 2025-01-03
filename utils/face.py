# -*- coding: utf-8 -*-
import json
from utils import common
from config import params

class FaceTool:
    '''
    获取已经训练书籍的信息
    '''

    def __init__(self):
        '''
        初始化
        '''
        self.face_config_json = params.face_config_json
        self.face_train_dir = params.face_train_dir
        # 配置face.json文件
        self.uniq_ids = []
        self.uniq_name_dict = {}
        self.all_face_infos = []
        self.all_names = []
        self.increment_uniq_faces = []
        self.get_current_all_face_info_from_config()
        if len(self.uniq_ids) > 0:
            self.max_uniq_id = str(int(max(self.uniq_ids)) + 1)
        else:
            self.max_uniq_id = '0'


    def fill_in_pinyin(self, json_data):
        """
        填充中文的拼音和英文翻译
        """
        name = json_data["name"]
        if name:
            #存在中文
            pinyin = json_data.get("pinyin")
            if pinyin:
                if pinyin != '':
                    print("不需要填充拼音")
                else:
                    json_data["pinyin"] = common.zh_to_pinyin(name)
            else:
                json_data["pinyin"] = common.zh_to_pinyin(name)
        else:
            print(f'JSON数据中 {json_data} 不存在该中文KEY bookName ')


    def get_current_all_face_info_from_config(self):
        '''
        读取配置文件获取所有人脸的信息， 主要是uniq_id
        '''
        with open(self.face_config_json, 'rb') as f:
            for line in f.readlines():
                line = line.decode('utf-8').replace('\n', '').replace('\r', '').replace('\'', '\"')
                #print(f'---------- {line}')
                if line != "":
                    one_face_json = json.loads(line)
                    uniq_id = one_face_json["uniq_id"]
                    name = one_face_json["name"]
                    is_train = one_face_json["is_train"]
                    if is_train == "False":
                        self.increment_uniq_faces.append(uniq_id)
                    self.uniq_name_dict[uniq_id] = name
                    self.fill_in_pinyin(one_face_json)
                    self.all_face_infos.append(one_face_json)
                    self.uniq_ids.append(uniq_id)
                    self.all_names.append(name)


    def get_increment_train_images(self):
        """
        获取所有训练人脸下面新增图像
        """
        images = []
        for increment_uniq in self.increment_uniq_faces:
            full_path = f'{self.face_train_dir}/{increment_uniq}'
            images = common.get_files_from_folder(full_path, images)
        return images


    def write_book_info_to_config(self):
        """
        将配置参数写入到文件中
        """
        with open(self.face_config_json, 'wb') as out_f:
            out_f.write(common.list_json_change_lines(self.all_face_infos).encode('utf-8'))


    def upadte_is_train_to_config(self, labels):
        """
        训练完后将is_train设置成True
        """
        for ele in self.all_face_infos:
            uniq_id = int(ele["uniq_id"])
            if uniq_id <= len(set(labels)):
                ele["is_train"] = "True"
        self.write_book_info_to_config()


    def zh_to_pinyin(self, face_names):
        '''
        将中文转成拼音
        '''
        results = []
        for ele in self.all_face_infos:
            name = ele['name']
            pinyin = ele['pinyin']
            if name in face_names:
                results.append(pinyin)
        return results

