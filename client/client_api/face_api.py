

def get_face(json_data):
    '''
    根据服务端返回的json解析最可能的人脸是谁
    '''
    results = []
    if json_data is not None:
        if 'face_recognition_result' in json_data.keys():
            face_resnet_label_results = json_data['face_recognition_result']
            # face_recognition_label_result = json_data['resnet_result']
            for k in face_resnet_label_results:
                results.append(k)
    return list(set(results))
