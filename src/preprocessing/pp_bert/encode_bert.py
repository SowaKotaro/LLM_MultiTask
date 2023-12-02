from transformers import BertJapaneseTokenizer

# def make_word_dict(text_list):
#     word2id = {'[pad]':0,'[s]':1,'[/s]':2,'[unk]':3}
#     num = 4
#     for text in text_list:
#         for word in text:
#             if not word in word2id:
#                 word2id[word] = num
#                 num += 1
#     id2word = {v: k for k,v in word2id.items()}
#     return word2id, id2word

# def change_base(morpheme):
#     full_seq = []
#     for m in morpheme:
#         word_data = m.split('\t\t')
#         base_seq = []
#         for wd in word_data:
#             base = wd.split('\t')[1].split(',')[7]
#             base_seq.append(base)
#         full_seq.append(base_seq)
#     return full_seq

# def change_seqid(wakati,word2id,frame_seqid,param):
#     seq_ids = []
#     for i,seq in enumerate(wakati):
#         seq_id = []
#         if param['cls']:
#             seq_id.append(word2id['[s]'])
#         for word in seq:
#             if word in word2id:
#                 seq_id.append(word2id[word])
#             else:
#                 seq_id.append(word2id['[unk]'])
#         if param['sep']:
#             seq_id.append(word2id['[/s]'])
#         num = -1
#         for j,same in enumerate(frame_seqid):
#             if i in same:
#                 num = j
#                 break
#         seq_ids.append([seq_id,num])
#     return seq_ids

# def seq_encode(wakati,morpheme,frame_seqid,param,word2id=None):
#     if param['base']:
#         wakati = change_base(morpheme)
#     if word2id is None:
#         word2id,id2word = make_word_dict(wakati)
#     seq_id = change_seqid(wakati,word2id,frame_seqid,param)
#     return seq_id,word2id,id2word

# def make_srl_dict(targets):
#     srl2id = {'[unk]':0}
#     frame_list = []
#     num = 1
#     for target in targets:
#         sl = []
#         for s,e,srl in target:
#             if not srl in frame_list and not srl in srl2id:
#                 if srl.isdecimal():
#                     frame_list.append(srl)
#                 else:
#                     srl2id[srl] = num
#                     num += 1
#                 #f not srl.isdecimal():
#                 #   srl_list.append(srl)
#     frame_list.sort(key=int)
#     first_frame_id = num
#     for frame in frame_list:
#        srl2id[frame] = num
#        num += 1
#     id2srl = {v: k for k,v in srl2id.items()}
#     #with open("./srl_list.txt",mode='w') as f:
#     #   f.writelines("\n".join(srl_list))
#     return srl2id,id2srl,first_frame_id

# def change_srlid(targets,srl2id):
#     for i,target in enumerate(targets):
#         for j,(_,_,srl) in enumerate(target):
#             if srl in srl2id:
#                 targets[i][j][2] = srl2id[srl]
#             else:
#                 targets[i][j][2] = srl2id['[unk]']
#     return targets

# def srl_encode(target,srl2id=None):
#     max_srl_id = -1
#     if srl2id is None:
#         srl2id,id2srl,first_frameid = make_srl_dict(target)
#     target_id = change_srlid(target,srl2id)
#     return target_id,srl2id,id2srl,first_frameid

# def make_framedic(wakati,target,morph):
#     frame_dic = {}
#     base_dic = {}
#     frame_seqid = {}
#     for i,(w,t,m) in enumerate(zip(wakati,target,morph)):
#         for s,e,srl in t:
#             if srl.isdecimal() and s != 0:
#                 base = ""
#                 for idx in range(s-1,e):
#                     base += m.split("\t\t")[idx].split('\t')[1].split(',')[7]
#                 #base = "".join(w[s-1:e])
#                 if not base in frame_dic:
#                     frame_dic[base] = [srl]
#                 elif not srl in frame_dic[base]:
#                     frame_dic[base].append(srl)
#                 if not "".join(w[s-1:e]) in base_dic:
#                     base_dic["".join(w[s-1:e])] = base
#                 if not base in frame_seqid:
#                     frame_seqid[base] = [i]
#                 else:
#                     frame_seqid[base].append(i)
#     #frame_dic = {key:val for key,val in frame_dic.items() if len(val)>=2}
#     with open("./frame_dict.txt",mode="w") as f:
#         for key,val in frame_dic.items():
#             print(key,val,file=f)
#     count = 0
#     new_frame_seqid = []
#     frame_key = []
#     for key,val in frame_dic.items():
#         if len(val) >= 2:
#             new_frame_seqid.append(frame_seqid[key])
#             frame_key.append([key,len(frame_seqid[key])])
#             count += 1
#     #print(len(frame_dic))
#     #print(count)
    
#     return frame_dic,base_dic,new_frame_seqid,frame_key


# def remove_frametag(targets,o_tag):
#     not_remove = ["[CLS]","[SEP]","O"]
#     for i in range(len(targets)):
#         index = targets[i].index(-1)
#         targets[i][index] = o_tag
#     return targets

def make_transition(tag2id,id2tag):
    transition = {}
    # cls = ["[CLS]"]
    # sep = ["[SEP]"]
    # o = ["O"]
    # print("tag2id [cls] is ",tag2id["[CLS]"])
    # print("tag2id [sep] is ",tag2id["[CLS]"])
    # print("tag2id",tag2id)
    cls = [tag2id["[CLS]"]]
    sep = [tag2id["[SEP]"]]
    # print("cls and sep ",cls,sep)
    o = [tag2id["O"]]
    b_tag = []
    i_tag = []
    # from
    for tag,id in tag2id.items(): # 各BIOタグについてB-のID,I-のIDでまとめる
        if tag == "[CLS]" or tag == "[SEP]" or tag == "O":
            continue
        bi = tag.split("-")[0] 
        arg = tag.split("-")[1:]
        arg = "-".join(arg)
        if bi == "B": # B-P B-意味役割のIDリスト
            #b_tag.append(tag)
            b_tag.append(id)
        if bi == "I": # I-P I-意味役割のIDリスト
            #i_tag.append(tag)
            i_tag.append(id)
    # to
    for tag,id in tag2id.items(): # それぞれのIDについて遷移がダメな設定をする
        if tag == "[CLS]":
            #transition[tag] = i_tag+sep+cls
            transition[id] = i_tag+sep+cls # clsから 各Iタグとsepとclsへは遷移できない
        elif tag == "[SEP]":
            #transition[tag] = i_tag+b_tag+cls+sep+o
            transition[id] = i_tag+b_tag+cls+sep+o # sepから すべてのタグに遷移できない
        elif tag == "O":
            #transition[tag] = i_tag+cls
            transition[id] = i_tag+cls # Oタグから 全てのIタグとclsには遷移できない
        else:
            bi = tag.split("-")[0]
            arg =  tag.split("-")[1:]
            arg = "-".join(arg)
            if bi == "B": # B-意味役割タグの時
                #transition[tag] = i_tag+cls
                transition[id] = i_tag+cls # 一旦,　全てのIタグとclsに遷移できない
                if not arg.isdecimal():    # 常にTrueになってIタグを作成する frameIdを使ってた時用の残骸
                    transition[id].remove(tag2id["I-{}".format(arg)]) # 自分の意味役割のIタグを除く
            if bi == "I": # I-意味役割タグの時
                transition[id] = i_tag+cls # 一旦,　全てのIタグとclsに遷移できない
                transition[id].remove(tag2id["I-{}".format(arg)]) # 自分のIタグを除く
    
    # print("transition",transition)
    return transition

def targetseq2id(targets,target_tags,predicate_not_O,predicate_bp):
    # print(targets)
    # print(target_tags)
    new_target = []
    #tag2id = {"[CLS]":0,"[SEP]":1,"O":2}
    tag2id = {"[CLS]":0,"[SEP]":1,"O":2,"B-P":3,"I-P":4}
    count = 5
    for tag in target_tags: # 意味役割の一覧リストからそれぞれのBタグとIタグの辞書を作成
        tag2id["B-{}".format(tag)] = count
        count += 1
        # tag2id["I-{}".format(tag)] = count  ## 通常処理
        # count += 1
        if not isinstance(tag, int): # 常にTrueになって意味役割のIタグ作成処理を実行 (FrameIDを含んでいたときの残骸)
            tag2id["I-{}".format(tag)] = count
            count += 1
    for h_idx,target in enumerate(targets): # データセットの文章でループ 出力系列のID化
        seq_index = []
        for i,word in enumerate(target): # 各単語について
            if not predicate_not_O: # 述語位置をOタグにする elseの方に行く
                #if word[2:].isdecimal():
                if word[2:].isdecimal() and word[0] == "B":
                    seq_index.append(tag2id["O"])
                    frame_position = i
                    continue
                elif  word[2:].isdecimal() and word[0] == "I":
                    seq_index.append(tag2id["O"])
                    # frame_position = i
                    continue
            else: # 述語位置をOタグ以外の場合
                if predicate_bp: # 述語位置を  B-P  I-P  にする  　ここを通る
                    if word[2:].isdecimal() and word[0] == "B": # 後半部分がFrameID(数値)かどうかで判定
                        seq_index.append(tag2id["B-P"])
                        frame_position = i
                        continue
                    elif word[2:].isdecimal() and word[0] == "I":
                        seq_index.append(tag2id["I-P"])
                        # frame_position = i
                        continue
                else: # 述語位置をB-FrameIDにする 
                    if word[2:].isdecimal() and word[0] == "B":
                        seq_index.append(tag2id[word])
                        frame_position = i
                        continue
                    elif word[2:].isdecimal() and word[0] == "I":
                        # seq_index.append(tag2id[word])  ## 通常処理
                        seq_index.append(tag2id["O"])  ## frameIdでIタグを作らない時の時用の処理 
                        # frame_position = i
                        continue

            if word in tag2id: # [CLS] [SEP] Oタグ B-意味役割 I-意味役割をID変換
                seq_index.append(tag2id[word])
            # else:
            #     if word[0] == "B":
            #         tag2id[word] = count
            #         count += 1
            #         seq_index.append(tag2id[word])
            #         srl = word.split("-")[1:]
            #         srl = "-".join(srl)
            #         tag2id["I-{}".format(srl)] = count
            #         count += 1
        new_target.append([seq_index,frame_position])
    id2tag = {v: k for k, v in tag2id.items()} # デコーダのためにタグとIDの辞書作成
    crf_transition = make_transition(tag2id,id2tag) # CRFで遷移可能かどうかの設定作成 B-Arg0からI-Arg0はOKだけど、I-Arg1には遷移できないような設定
    return new_target,id2tag,crf_transition

def seq2bertid(wakati,dict_path): # Unidic辞書でBERTに対応したID変換
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2',mecab_kwargs={"mecab_option": "-d{}".format(dict_path)})
    # tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    # tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small")
    # print("wakati",wakati[0])
    wakati_id = []
    for seq in wakati:
        wakati_id.append(tokenizer.convert_tokens_to_ids(seq))

    return wakati_id

def targetseq2id_fidin(targets,target_tags,predicate_not_o,predicate_bp,frame_list=[]):
    new_target = []
    #tag2id = {"[CLS]":0,"[SEP]":1,"O":2}
    tag2id = {"[CLS]":0,"[SEP]":1,"O":2,"B-P":3,"I-P":4}
    count = 5
    for tag in target_tags:
        tag2id["B-{}".format(tag)] = count
        count += 1
        # tag2id["I-{}".format(tag)] = count  ## 通常処理
        # count += 1
        if not isinstance(tag, int): ## frameIdでIタグを作らない時の時用の処理
            tag2id["I-{}".format(tag)] = count
            count += 1
    for h_idx,target in enumerate(targets):
        seq_index = []
        for i,word in enumerate(target):
            if not predicate_not_o:
                #if word[2:].isdecimal():
                if word[2:].isdecimal() and word[0] == "B":
                    seq_index.append(tag2id["O"])
                    frame_position = i
                    continue
                elif  word[2:].isdecimal() and word[0] == "I":
                    seq_index.append(tag2id["O"])
                    # frame_position = i
                    continue
            else:
                if predicate_bp:
                    if word[2:].isdecimal() and word[0] == "B":
                        seq_index.append(tag2id["B-P"])
                        frame_position = i
                        frame_id = frame_list.index(int(word[2:]))
                        continue
                    elif word[2:].isdecimal() and word[0] == "I":
                        seq_index.append(tag2id["I-P"])
                        # frame_position = i
                        continue
                else:
                    if word[2:].isdecimal() and word[0] == "B":
                        seq_index.append(tag2id[word])
                        frame_position = i
                        continue
                    elif word[2:].isdecimal() and word[0] == "I":
                        # seq_index.append(tag2id[word])  ## 通常処理
                        seq_index.append(tag2id["O"])  ## frameIdでIタグを作らない時の時用の処理 
                        # frame_position = i
                        continue

            if word in tag2id:
                seq_index.append(tag2id[word])
            # else:
            #     if word[0] == "B":
            #         tag2id[word] = count
            #         count += 1
            #         seq_index.append(tag2id[word])
            #         srl = word.split("-")[1:]
            #         srl = "-".join(srl)
            #         tag2id["I-{}".format(srl)] = count
            #         count += 1
        new_target.append([seq_index,frame_position,frame_id])
    id2tag = {v: k for k, v in tag2id.items()}
    crf_transition = make_transition(tag2id,id2tag)
    return new_target,id2tag,crf_transition

def get_sentenceid(json_list):
    sentence_id_list = []
    for json_ in json_list:
        sentence_id_list.append(json_["sentenceID"])
    return sentence_id_list
  
def target2id(targets, target_tags):
    target_id = []
    for fid in targets:
        target_id.append(target_tags.index(fid))
    return target_id

def target2id_v2(targets, target_tags, target_candidate):
    target_id = []
    target_candidate_id = []
    for fid,candidate in zip(targets,target_candidate): # 正解と予測候補それぞれ　ID変換
        target_id.append(target_tags.index(fid)) # 正解FrameIDをIDに変換
        if candidate == [-1]: # [-1]のとき、候補が決まってないからそのまま(後にすべてのFrameIDを候補として扱う)
            target_candidate_id.append(candidate)
        else:
            target_candidate_id.append([target_tags.index(i) for i in candidate]) # 正解候補FrameIDをID変換
    return target_id,target_candidate_id