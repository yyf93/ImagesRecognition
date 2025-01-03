# -*- coding: utf-8 -*-
import json
import os
import datetime
from config import params

class FileUtils:
    '''
    处理文件的工具
    '''
    @staticmethod
    def get_files_from_folder(dir, all_list):
        """
        获取单个目录下所有的文件
        """
        for item in os.listdir(dir):
            full_path = os.path.join(dir, item)
            if os.path.isfile(full_path):
                all_list.append(full_path.replace('\\', '/'))
        return all_list

    @staticmethod
    def get_subdirs(dir):
        """
        列出所有子目录名称
        """
        subs = []
        for item in os.listdir(dir):
            full_path = os.path.join(dir, item)
            if os.path.isdir(full_path):
                subs.append(full_path.replace('\\', '/'))
        return subs

    @staticmethod
    def save_requests_file(request):
        """
        返回接收到请求的文件路径， 并保存该文件
        """
        file = request.files['image']
        tmp_path = params.base_directory + '/' + file.filename
        file.save(tmp_path)
        print(f"receive img: {tmp_path}")
        return tmp_path

    @staticmethod
    def is_exit_file(filename):
        """
        如果存在文件
        """
        if os.path.exists(filename):
            print("Path exists")
            return True
        else:
            print("Path does not exist")
            return False

    @staticmethod
    def delete_file(file_path):
        """
        要删除的文件路径
        """
        try:
            # 删除文件
            os.remove(file_path)
            print(f'文件 {file_path} 已删除')
        except FileNotFoundError:
            print(f'文件 {file_path} 不存在')
        except PermissionError:
            print(f'没有权限删除文件 {file_path}')
        except Exception as e:
            print(f'发生错误: {e}')


class EncryptionUtils:
    '''
    加解密的工具
    '''
    @staticmethod
    def aes_encrypt(key, input_bytes):
        """
        将字节流通过AES加密
        """
        from Crypto.Cipher import AES
        import base64
        try:
            cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
            block_size = AES.block_size
            pad = lambda s: s + (block_size - len(s) % block_size) * bytes([block_size - len(s) % block_size])
            input_bytes = pad(input_bytes)
            encrypted_bytes = cipher.encrypt(input_bytes)
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def aes_decrypt(key, encrypted_text):
        """
        将字节流通过AES解密
        """
        from Crypto.Cipher import AES
        import base64
        try:
            cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
            encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
            decrypted_bytes = cipher.decrypt(encrypted_bytes)
            unpad = lambda s: s[0:-s[-1]]
            return unpad(decrypted_bytes)
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def decode_save_picture(compressed_data, file_name):
        """
        将字节流解密并保存成图片
        """
        # decompressed_data = zlib.decompress(compressed_data).decode('utf-8')
        decompressed_data = compressed_data.decode('utf-8')
        key = params.scert_aes_key
        pic_bytes = EncryptionUtils.aes_decrypt(key, decompressed_data)
        # file_name = 'C:/Users/Administrator/Desktop/tmp/aes_people_tmp_20230613151233.jpg'
        with open(file_name, 'wb') as f:
            f.write(pic_bytes)


class EmailUtils:
    '''
    邮件工具
    '''

    @staticmethod
    def send_email(title, data, attachment_file_name):
        """
        发送邮件， 请传入标题， 内容，附件名
        """
        import smtplib
        from email.message import EmailMessage

        mail_host = params.mail_host
        mail_user = params.mail_user
        mail_pass = params.mail_pass
        mail_port = params.mail_port
        sender = params.mail_sender
        receivers = params.mail_receivers

        subject = title
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = receivers

        # 设置邮件正文
        msg.set_content('see attachment')

        # 添加附件
        # if action == "borrow" or action == "return":
        #     flag_flow, file_data, file_name = book_messages.get_borrow_return_json_file(img_file, action, book_id, book_name)
        # elif action == "add":
        #     flag_flow, file_data, file_name = book_messages.get_add_json_file(img_file, action)
        # print(file_data)
        # if flag_flow == False:
        #     return file_data
        # print(file_name)
        msg.add_attachment(data, filename=attachment_file_name)
        try:
            # 连接SMTP服务器并登录
            smtp_server = smtplib.SMTP_SSL(mail_host, mail_port)
            smtp_server.login(mail_user, mail_pass)
            # 发送邮件
            smtp_server.send_message(msg)
            smtp_server.quit()
            return True, ""
        except Exception as e:
            print(e)
            return False, e


class MD5Utils:
    '''
    MD5工具
    '''
    @staticmethod
    def md5_encrypt(data):
        '''
        获取字符串的MD5
        '''
        import hashlib
        md5_str = hashlib.md5()
        md5_str.update(data.encode('utf-8'))
        return md5_str.hexdigest()


class TranslationUtils:
    '''
    翻译工具
    '''
    @staticmethod
    def zh_to_pinyin(zh_str):
        '''
        中文转拼音
        '''
        import pypinyin
        # 默认风格,包含音标
        pinyin = pypinyin.pinyin(zh_str, style=pypinyin.NORMAL)
        return ''.join(sum(pinyin, []))

    @staticmethod
    def zh_to_en(zh_str):
        '''
        中文转英文
        '''
        import requests
        result = None
        q = zh_str
        appid = params.baidu_api_fanyi_appid
        salt = params.baidu_api_fanyi_salt
        scret_key = params.baidu_api_fanyi_scret_key
        sign_pre = f'{appid}{q}{salt}{scret_key}'
        sign = MD5Utils.md5_encrypt(sign_pre)
        try:
            url = f"http://api.fanyi.baidu.com/api/trans/vip/translate?q={q}&from=zh&to=en&appid={appid}&salt={salt}&sign={sign}"
            r = requests.get(url)
            result_json = r.json()
            result = result_json["trans_result"][0].get("dst")
        except Exception as e:
            print(e)
        return result





def get_key_from_json(json_data):
    '''
    从json中获取keys
    '''
    keys = []
    for key in json_data.keys():
        keys.append(key)
    return keys


def get_current_time_str():
    """
    获取当前时间
    """
    now = datetime.datetime.now()
    timestamp = now.timestamp()
    time_format = '%Y%m%d%H%M%S'
    return datetime.datetime.fromtimestamp(timestamp).strftime(time_format)


def get_trains_sub_folder_files(path, images, is_enhance=False):
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
        for dir in dirs:
            print("==========", root, dir)
            files_and_folders = os.listdir(root+'/'+dir)
            # 过滤出当前工作目录下的所有文件
            files_only = [f for f in files_and_folders if os.path.isfile(os.path.join(root+'/'+dir, f))]
            #判断如果文件数超过多少， 就不进行enchance
            if len(files_only) >= 25 and is_enhance:
                print(f"单个图片素材数量为{len(files_only)}, 超过25张图, 不再进行图像增强")
                continue
            for filename in files_only:
                full_path = root + '/' + dir + '/' + filename
                #print('---------------------', full_path)
                images.append(full_path)
    return images


def reverse_dict(dict):
    """
    反转字典
    """
    new_dict = {}
    for k, v in dict.items():
        new_dict[v] = k
    return new_dict


def list_json_change_lines(list_json_info):
    '''
    将list的json转成line, 用\n去分割
    '''
    lines = ''
    for ele in list_json_info:
        json_str = json.dumps(ele, ensure_ascii=False)
        line = f'{json_str}\n'
        lines += line
    return lines














if __name__ == '__main__':
    print(zh_to_pinyin("Dunkirk: The History Behind the Major Motion Picture"))
    # get_files_and_folder('./images/train_books', [])
    # print(get_subdirs('./images/train_books'))
    # print(get_files_from_folder('./images/train_books/9787506394314', []))
