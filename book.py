# -*- coding: utf-8 -*-
import json
import common
import params


class BookTool:
    '''
    获取已经训练书籍的信息
    '''

    def __init__(self):
        '''
        初始化
        '''
        self.book_config_file = params.book_config_json
        self.book_train_dir = params.book_train_dir
        # 用于写入配置book.json文件
        self.all_book_infos = []
        self.uniq_ids = []
        self.isbns = []
        self.current_book_isbn_id_json = {}
        self.book_name_id = {}
        self.book_info_json = {}
        self.book_train_dict = {}
        self.get_current_all_book_info_from_config()
        self.increment_book_isbns = []
        if len(self.uniq_ids)  > 0:
            self.max_uniq_id = str(int(max(self.uniq_ids)) + 1)
        else:
            self.max_uniq_id = '0'


    def is_exit_isbn(self, new_isbn):
        """
        判断是否存在该isbn,在已经有的配置中
        """
        print(self.current_book_isbn_id_json.keys())
        if new_isbn in self.current_book_isbn_id_json.keys():
            return True
        else:
            return False


    def get_book_uniq_id_by_name(self, book_name):
        """
        通过图书名称获取他的uniq_id
        """
        if book_name in self.book_name_id.keys():
            return self.book_name_id[book_name]
        else:
            return None


    def get_book_info_from_api(self, isbn):
        '''
        获取单个书籍信息
        '''
        import requests
        headers = {
            'Host': 'bg.xdhtxt.com',
            'Connection': 'keep-alive',
            'xweb_xhr': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 MicroMessenger/7.0.20.1781(0x6700143B) NetType/WIFI MiniProgramEnv/Windows WindowsWechat/WMPF XWEB/8391',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'Referer': 'https://servicewechat.com/wx7daa8a3d8882c4ed/4/page-frame.html',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh',
        }
        response = requests.get(f'https://bg.xdhtxt.com/isbn/json/{isbn}?token={params.isbn_token}', headers=headers)
        try:
            json_data = json.loads(response.text)
            book_info_json = json_data["result"]
            # clean api json
            for k, v in book_info_json.items():
                if v is None:
                    book_info_json[k] = ''
                if k == 'pictures':
                    book_info_json[k] = json.loads(v)
                elif k == 'bookName':
                    book_info_json[k] = v.replace(':', '').split('(')[0]
            book_info_json['uniq_id'] = self.max_uniq_id
            book_isbn = book_info_json['isbn']
            book_name = book_info_json['bookName']
            book_info_json['is_train'] = "False"
            self.book_name_id[book_name] = self.max_uniq_id
            self.all_book_infos.append(book_info_json)
            self.current_book_isbn_id_json[book_isbn] = self.max_uniq_id
            self.max_uniq_id = str(int(max(self.max_uniq_id)) + 1)
            return True
        except Exception as e:
            print(e)
            return False


    def get_book_train_info(self):
        '''
        获取该书籍是否已经被训练过
        '''
        return self.book_train_dict


    def get_current_all_book_info_from_config(self):
        '''
        读取配置文件获取所有书籍的信息， 主要是isbn+uniq_id
        '''
        with open(self.book_config_file, 'rb') as f:
            for line in f.readlines():
                line = line.decode('utf-8').replace('\n', '').replace('\r', '').replace('\'', '\"')
                #print(f'---------- {line}')
                if line != "":
                    one_book_json = json.loads(line)
                    uniq_id = one_book_json["uniq_id"]
                    isbn = one_book_json["isbn"]
                    name = one_book_json["bookName"]
                    is_train = one_book_json["is_train"]
                    self.uniq_ids.append(uniq_id)
                    self.isbns.append(isbn)
                    self.all_book_infos.append(one_book_json)
                    self.current_book_isbn_id_json[isbn] = uniq_id
                    self.book_name_id[name] = uniq_id
                    self.book_train_dict[isbn] = is_train


    def get_uniq_id(self, isbn):
        '''
        通过isbn, 获取图书的uniq_id
        '''
        uniq_id = self.current_book_isbn_id_json[isbn]
        if uniq_id:
            return uniq_id
        else:
            print(f"该isbn: {isbn} 尚未通过API获取该信息并写入到./config/books.json中, 请先进行训练")
            return None


    def write_book_info_to_config(self):
        """
        将所有的信息写到配置文件中
        """
        with open(self.book_config_file, 'wb') as out_f:
            out_f.write(common.list_json_change_lines(self.all_book_infos).encode('utf-8'))


    def get_all_isbns_from_train_dir(self):
        '''
        获取训练路径下的isbn， 如果没有和配置文件匹配上， 将会新增到配置文件上！
        '''
        not_in_config_isbn = []
        isbn_dirs = common.get_subdirs(self.book_train_dir)
        # 警告未拍摄的图像，但是存在于配置文件中， 发送警告信息， 让管理员进行图像训练
        self.warning_not_capture_books(isbn_dirs)
        for one_dir_name in isbn_dirs:
            train_isbn = one_dir_name.split('/')[-1]
            if train_isbn not in self.isbns:
                not_in_config_isbn.append(train_isbn)
        self.increment_book_isbns = not_in_config_isbn
        if len(not_in_config_isbn) > 0:
            print(f'increment isbns {not_in_config_isbn}')
            #如果有新增的书籍图书， 将会将新的图书信息放到书籍的配置文件中
            for new_isbn in not_in_config_isbn:
                flag = self.get_book_info_from_api(new_isbn)
                if flag:
                    print("获取新增书籍信息成功")
                else:
                    print("获取新增书籍信息失败")
            self.write_book_info_to_config()
        return not_in_config_isbn


    def warning_not_capture_books(self, isbn_dirs):
        """
        警告未拍摄的图像，但是存在于配置文件中， 发送警告信息， 让管理员进行图像训练
        """
        dir_isbns = []
        more_isbns = []
        if len(isbn_dirs) < len(self.isbns):
            for one_dir_name in isbn_dirs:
                train_isbn = one_dir_name.split('/')[-1]
                dir_isbns.append(train_isbn)
            for ele in self.isbns:
                if ele not in dir_isbns:
                    more_isbns.append(ele)
        print(f"WARNING: {more_isbns}未拍摄图像并进行训练！！！")


    def upadte_is_train_to_config(self, train_labels):
        """
        跟新下配置文件中的is_train状态
        """
        for ele in self.all_book_infos:
            uniq_id = int(ele["uniq_id"])
            if uniq_id <= len(set(train_labels)):
                ele["is_train"] = "True"
        self.write_book_info_to_config()


    def is_increment_book(self):
        """
        判断是否为新增图书
        """
        flag = False
        # 检查是否需要有新增书籍但是没有拍摄样本的情况
        train_isbns = self.get_all_isbns_from_train_dir()
        if len(train_isbns) > 0:
            flag = True
        return flag


    def get_increment_train_images(self):
        """
        获取所有训练书籍下面新增书籍图像
        """
        images = []
        if self.is_increment_book():
            for increment_isbn in self.increment_book_isbns:
                full_path = f'{self.book_train_dir}/{increment_isbn}'
                images = common.get_files_from_folder(full_path, images)
        return images


    def get_borrow_return_json_file(self, img_file, action, book_name):
        """
        通过传入参数图片地址,借还书类型,书ID,书名去进行产生借还书的JSON
        """
        try:
            current_date = common.get_current_time_str()
            # get and validate book_id 获取图书的UNIQ_ID, 如果是新增书，还未收录的情况，返回-1
            book_id = self.get_book_uniq_id_by_name(book_name)
            if book_id is None:
                book_id = -1
            # params
            key = params.scert_aes_key
            # process_one_picture
            # read_file = 'C:/Users/Administrator/Desktop/tmp/people_tmp_20230613151233.jpg'
            with open(img_file, "rb") as f:
                input_bytes = f.read()
            encrypted_text = common.aes_encrypt(key, input_bytes)
            # print(encrypted_text)
            # decrypted_data = aes_decrypt(key, encrypted_text)
            # compressed_zlib = zlib.compress(encrypted_text.encode('utf-8')).decode('utf-8')
            # book_id = 123
            # AI借书ID都是99
            user_id = 99
            # action = 'borrow'
            # book_name = 'X'
            img_json = f" \"imgBytes\": \"{encrypted_text}\" "
            current_json = f" \"updateTime\": {current_date} "
            book_id_json = f" \"bookId\": {book_id} "
            book_name_json = f" \"bookName\": \"{book_name}\" "
            user_json = f" \"userId\": {user_id} "
            action_json = f" \"action\": \"{action}\" "
            one_json = "{" + book_id_json + ", " + current_json + ', ' + book_name_json + ", " + action_json + ", " + user_json + ", " + img_json + " }\n"
            # print(one_json)
            # decode_save_picture(compressed_zlib)
            return one_json
        except Exception as e:
            print(e)
            return None


    def extract_numbers_from_image_by_orc(self, image):
        """
        通过orc提取图中数字
        """
        import pytesseract
        # 使用pytesseract识别图像中的文本
        extracted_text = pytesseract.image_to_string(image)
        # 从提取的文本中筛选出数字
        extracted_numbers = ''.join(filter(str.isdigit, extracted_text))
        if len(extracted_numbers) >= 13:
            print(f'isbn orc extract: {extracted_numbers[:13]}')
            return extracted_numbers[:13]
        else:
            return None


    def extract_numbers_from_image_by_pyzbar(self, image):
        """
        通过pyzbar提取图中条形码
        """
        from pyzbar.pyzbar import decode
        try:
            # 解码图像
            decode_data = decode(image)
            if decode_data:
                if len(decode_data) >= 1:
                    print(f'isbn pyzbar extract: {decode_data[0].data}')
                    return int(decode_data[0].data)
        except Exception as e:
            print(e)
            print("pyzbar can't read barcode")
        return None


    def get_isbn_from_img(self, img_file):
        """
        通过orc和条形码库去获取书籍的isbn
        """
        import cv2
        # 读取图像
        img = cv2.imread(img_file)
        num = self.extract_numbers_from_image_by_pyzbar(img)
        if num:
            return num
        else:
            #num = self.extract_numbers_from_image_by_orc(img)
            return None


    def get_add_json_file(self, img_back_file, isbn=None):
        """
        返回添加书籍内容的json,
        TODO: 如果成功识别到图书， 则需要将书的封面保存到train_books中， 方便下次的训练, 存在一个问题摄像头拍摄的存在问题
        """
        action = "add"
        one_json = None
        current_date = common.get_current_time_str()
        # 如果传入的参数isbn长度不对或者直接为空则通过图片获取条形码isbn
        if (isbn and len(str(isbn)) < 13) or (isbn is None ) and img_back_file is not None:
            isbn = self.get_isbn_from_img(img_back_file)
        if isbn:
            if len(str(isbn)) >= 13:
                isbn = str(isbn)[:13]
            if self.is_exit_isbn(isbn):
                print("已经存在这个isbn， 不需要新增")
                return None
            else:
                # 通过API获取图书信息， 放在最后一个all_book_infos
                info_flag = self.get_book_info_from_api(isbn)
                if info_flag:
                    book_info_json = self.all_book_infos[-1]
                    id = book_info_json['uniq_id']
                    name = book_info_json["bookName"]
                    author = book_info_json["author"]
                    publisher = book_info_json["press"]
                    date_published = book_info_json["pressDate"]
                    details = book_info_json["bookDesc"]
                    book_category = book_info_json["categoryName"]
                    id_json = f" \"id\": {id} "
                    isbn_json = f" \"isbn\": \"{isbn}\" "
                    action_json = f" \"action\": \"{action}\" "
                    name_json = f" \"name\": \"{name}\" "
                    current_json = f" \"updateTime\": \"{current_date}\" "
                    author_json = f" \"author\": \"{author}\" "
                    publisher_json = f" \"publisher\": \"{publisher}\" "
                    date_published_json = f" \"datePublished\": \"{date_published}\" "
                    details_json = f" \"details\": \"{details}\" "
                    book_category_json = f" \"bookCategory\": \"{book_category}\" "
                    quantity_json = " \"quantity\": 1 "
                    status_json = " \"status\": \"已入库\" "
                    location_json = " \"location\": \"花木17F\" "
                    one_json = "{ " + id_json + ", " + isbn_json + ", " + action_json + ", " + location_json + ", " + status_json + ", " + quantity_json + ", " + book_category_json + ", " + name_json + ", " + current_json + ', ' + author_json + ", " + publisher_json + ", " + date_published_json + ", " + details_json + " }\n"
                    self.write_book_info_to_config()
                else:
                    print("不能获取图书网页的API信息")
        else:
            print("无法识别图像中的条形码")
        return one_json


if __name__ == '__main__':
    book_tool = BookTool()
    # print(book_tool.is_increment_book())
    print(book_tool.current_book_isbn_id_json)
    print(book_tool.get_increment_train_images())
    print(book_tool.current_book_isbn_id_json)