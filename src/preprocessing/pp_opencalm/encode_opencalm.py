from transformers import AutoTokenizer

def make_transition_opencalm(tag2id,id2tag):
    transition = {}
    # cls = [tag2id["[CLS]"]]
    # sep = [tag2id["[SEP]"]]
    # print("cls and sep ",cls,sep)
    eot = [tag2id["<|endoftext|>"]]
    o = [tag2id["O"]]
    b_tag = []
    i_tag = []
    add_i_tag = []
    # from
    for tag,id in tag2id.items(): # 各BIOタグについてB-のID,I-のIDでまとめる
        if tag == "<|endoftext|>" or tag == "O":
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
        # print("tag",tag,"\nid",id,"\ni_tag",i_tag)
        # print("i_tag",i_tag,"\nall_itag",all_i_tag)
        if tag == "<|endoftext|>":
            #transition[tag] = i_tag+sep+cls
            transition[id] = i_tag+eot # clsから 各Iタグとsepとclsへは遷移できない
        # elif tag == "[SEP]":
        #     #transition[tag] = i_tag+b_tag+cls+sep+o
        #     transition[id] = i_tag+b_tag+eot+o # sepから すべてのタグに遷移できない
        elif tag == "O":
            #transition[tag] = i_tag+cls
            transition[id] = i_tag # Oタグから 全てのIタグとclsには遷移できない
        else:
            # print("aaa",i_tag)
            bi = tag.split("-")[0]
            arg =  tag.split("-")[1:]
            arg = "-".join(arg)
            if bi == "B": # B-意味役割タグの時
                # print("bbb",i_tag)
                #transition[tag] = i_tag+cls
                transition[id] = i_tag # 一旦,　全てのIタグとclsに遷移できない
                # if not arg.isdecimal():    # 常にTrueになってIタグを作成する frameIdを使ってた時用の残骸
                if not arg.isdecimal(): 
                    transition[id].remove(tag2id["I-{}".format(arg)]) # 自分の意味役割のIタグを除く
                    # add_i_tag.append(tag2id["I-{}".format(arg)])
                
                    # i_tag = add_i_tag + i_tag
                    i_tag = i_tag + [id+1]
                    # print("b-b",i_tag)
            elif bi == "I": # I-意味役割タグの時  
                # print("ccc",i_tag)
                transition[id] = i_tag # 一旦,　全てのIタグとclsに遷移できない
                # transition[id].remove(id)
                transition[id].remove(tag2id["I-{}".format(arg)]) # 自分のIタグを除く
                # add_i_tag.append(tag2id["I-{}".format(arg)])
                # i_tag.append(add_i_tag)
                i_tag = i_tag + [id]
                # print("c-c",i_tag)
    # print("transition[54]",transition[54])
    # print("transition[55]",transition[55])
    
    # print("YATTA!")
    return transition

def targetseq2id_opencalm(targets,target_tags,predicate_not_O,predicate_bp):
    # print(targets)
    # print(target_tags)
    new_target = []
    #tag2id = {"[CLS]":0,"[SEP]":1,"O":2}
    # tag2id = {"[CLS]":0,"[SEP]":1,"O":2,"B-P":3,"I-P":4}
    tag2id = {"<|endoftext|>":0,"O":1,"B-P":2,"I-P":3}
    count = 4
    for tag in target_tags: # 意味役割の一覧リストからそれぞれのBタグとIタグの辞書を作成
        tag2id["B-{}".format(tag)] = count
        count += 1
        # tag2id["I-{}".format(tag)] = count  ## 通常処理
        # count += 1
        if not isinstance(tag, int): # 常にTrueになって意味役割のIタグ作成処理を実行 (FrameIDを含んでいたときの残骸)
            tag2id["I-{}".format(tag)] = count
            count += 1
    # print("tags", tag2id)
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
    crf_transition = make_transition_opencalm(tag2id,id2tag) # CRFで遷移可能かどうかの設定作成 B-Arg0からI-Arg0はOKだけど、I-Arg1には遷移できないような設定
    return new_target,id2tag,crf_transition

def seq2id_opencalm(wakati,dict_path, MODEL): # TokenizerでToken IDに変換
    if "SMALL" in MODEL:
        tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small")
    elif "MEDIUM" in MODEL:
        tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-medium")

    wakati_id = []
    # token_head_id = []
    for seq in wakati:
        token_head_id = []
        for token in seq:
            encode_id = tokenizer.encode(token)
            token_head_id.append(encode_id[0])  # token_idの先頭だけを取り出す
        # wakati_id.append(tokenizer.encode(token)) # トークンごとにエンコード
        wakati_id.append(token_head_id)
    return wakati_id
    # return token_head_id

def targetseq2id_fidin_opencalm(targets,target_tags,predicate_not_o,predicate_bp,frame_list=[]):
    new_target = []
    #tag2id = {"[CLS]":0,"[SEP]":1,"O":2}
    # tag2id = {"[CLS]":0,"[SEP]":1,"O":2,"B-P":3,"I-P":4}
    tag2id = {"<|endoftext|>":0,"O":1,"B-P":2,"I-P":3}
    count = 4
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
    crf_transition = make_transition_opencalm(tag2id,id2tag)
    return new_target,id2tag,crf_transition

def get_sentenceid_opencalm(json_list):
    sentence_id_list = []
    for json_ in json_list:
        sentence_id_list.append(json_["sentenceID"])
    return sentence_id_list
  
def target2id_opencalm(targets, target_tags):
    target_id = []
    for fid in targets:
        target_id.append(target_tags.index(fid))
    return target_id

def target2id_v2_opencalm(targets, target_tags, target_candidate):
    target_id = []
    target_candidate_id = []
    for fid,candidate in zip(targets,target_candidate): # 正解と予測候補それぞれ　ID変換
        target_id.append(target_tags.index(fid)) # 正解FrameIDをIDに変換
        if candidate == [-1]: # [-1]のとき、候補が決まってないからそのまま(後にすべてのFrameIDを候補として扱う)
            target_candidate_id.append(candidate)
        else:
            target_candidate_id.append([target_tags.index(i) for i in candidate]) # 正解候補FrameIDをID変換
    return target_id,target_candidate_id