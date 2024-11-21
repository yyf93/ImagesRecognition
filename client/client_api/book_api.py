


def get_book(json_data):
    '''
    根据服务端返回的json解析最可能的图书是谁
    '''
    results = []
    #sklearn_models = ['svm']
    if json_data is not None:
        #sklearn_result = json_data['sklearn_result']
        resnet_result = json_data['resnet_result']
        for k, v in resnet_result.items():
            results.append(k)
        #for k, v in sklearn_result.items():
            #if k in sklearn_models:
                #results.append(v)
    return list(set(results))



