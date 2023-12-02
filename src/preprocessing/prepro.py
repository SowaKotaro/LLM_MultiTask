import json,re
#from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
import random

def read_params(path):
    # print(path)
    with open('./../data/{}.txt'.format(path),mode='r') as f:
        text = f.read()
    text = re.sub(r'#.*','',text)
    json_ = json.loads(text)
    for key in json_.keys():
        if json_[key] == 'False' or json_[key] == 'false':
            json_[key] = False
        if json_[key] == 'True' or json_[key] == 'true':
            json_[key] = True
    return json_


# def write_ids(wakati_id,target_id,word2id,srl2id):
#     tmp = []
#     for wa in wakati_id:
#         tmp.append(' '.join(str(wa[0]))+' '+str(wa[1]))
#     tar = []
#     for ta in target_id:
#         str_ = ''
#         for t in ta:
#             str_ += ' '.join(str(t))
#         tar.append(str_)
#     with open('wakati_id.txt',mode='w') as f:
#         f.writelines("\n".join(tmp))
#     with open('target_id.txt',mode='w') as f:
#         f.writelines("\n".join(tar))
#     #with open('word2id.json',mode='w') as f:
#     #    json.dump(word2id,f,ensure_ascii=False)
#     #with open('srl2id.json',mode='w') as f:
#     #    json.dump(srl2id,f,ensure_ascii=False)

def data_split(x,y,shuffle,rate):
    if shuffle:
        x_train,x_test,y_train,t_test = train_test_split(x,y,shuffle=shuffle,random_state=0,test_size=rate)
    else:
        x_train,x_test,y_train,t_test = train_test_split(x,y,shuffle=shuffle,test_size=rate)

    return x_train,x_test,y_train,t_test

def sentence_shuffle(json_list):
    random.seed(0)
    sentence_dict = {}
    id_list = []
    new_json_list = []
    for i,json in enumerate(json_list):
        if json["sentenceID"] in sentence_dict.keys():
            sentence_dict[json["sentenceID"]].append(i)
        else:
            sentence_dict[json["sentenceID"]] = [i]
            id_list.append(json["sentenceID"])
    # print(id_list[:5])
    random.shuffle(id_list)
    # print("文章数:",len(id_list))
    # print("述語数:",len(json_list))
    # print(id_list[:5])
    for id in id_list:
        for index in sentence_dict[id]:
            new_json_list.append(json_list[index])
    # print(len(json_list))
    # print(len(new_json_list))
    # print(json_list[:3])
    # print(new_json_list[:3])
    return new_json_list

def frameid2tag():
    with open('./../data/all_fid.txt', mode='r') as f: #  辞書のFrameID列の部分
        fid_list = f.readlines()
    fid_list = [int(i.replace("\n","")) for i in fid_list if i != "\n"]
    fid_list = list(set(fid_list)) # 重複なしFrameIDの一覧リスト
    # fid_list.append(2029)
    # fid_list.append(0)
    # fid_list.append(1095)
    fid_list.sort()
    return fid_list

"""
def data_split_v2(x, y, z, shuffle, rate):
    if shuffle:
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
            x, y, z, shuffle=shuffle, random_state=0, test_size=rate)
    else:
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
            x, y, z, shuffle=shuffle, test_size=rate)

    return x_train, x_test, y_train, y_test, z_train, z_test
"""