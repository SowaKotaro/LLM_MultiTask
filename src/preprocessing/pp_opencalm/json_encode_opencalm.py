def make_bioseq_opencalm(json,cls,sep,arg_role,frame_point):
    # print("make_bioseq() in json_encode.py")
    #target = ["O"]*(json["length"]+cls+sep) #json["length"]
    tags = []
    target = ["O"]*(len(json["sentence"].split())+cls+sep) # 出力系列分のOタグを作成
    if cls: # 先頭にcls
        target[0] = "<|endoftext|>"
    if sep: # 末尾にsep
        target[-1] = "<|endoftext|>"
    
    for args in json["args"]: # 意味役割でループを回す
        if args["word_start"] != -1: # 文章で省略されている意味役割のword_startは[-1]なので　それ以外の場合にTrue
            #target[args["char_start"]] = "B-{}".format(args[arg_role])
            target[args["word_start"]+1] = "B-{}".format(args[arg_role]) # 現在の意味役割の開始位置に"B-意味役割"タグを出力系列に代入  +1はclsの分
            #target[args["char_start"]+1:args["char_end"]+1] = ["I-{}".format(args[arg_role])]*(args["char_end"]-args["char_start"])
            target[args["word_start"]+1+1:args["word_end"]+1+1] = ["I-{}".format(args[arg_role])]*(args["word_end"]-args["word_start"]) # 意味役割が2単語数分以上ある場合は"I-意味役割"タグを代入  +1+1はそれぞれclsの分とB-意味役割の分
            tags.append(args[arg_role]) # 意味役割を追加,後で全ての意味役割のリストを作るため
    
    if frame_point: # 述語位置用の処理
        frame = json["predicate"]
        # frame_id = int(frame["frameID"].split(":")[0])
        frame_id = frame["frameID"]
        #if frame["char_start"] == frame["char_end"]:
        #    target[frame["char_start"]] = "B-{}".format(frame["frame"])
        if frame["word_start"] == frame["word_end"]: # 述語が1単語の時
            target[frame["word_start"]+1] = "B-{}".format(frame_id) # 述語位置に"B-FrameID"形式で出力系列に代入 "後に"B-P"にする
        else: # 述語が2単語以上の時
            #target[frame["char_start"]] = "B-{}".format(frame["frame"])
            #target[frame["char_start"]+1:frame["char_end"]+1] = ["I-{}".format(frame["frame"])]*(frame["char_end"]-frame["char_start"])
            target[frame["word_start"]+1] = "B-{}".format(frame_id)
            target[frame["word_start"]+1+1:frame["word_end"]+1+1] = ["I-{}".format(frame_id)]*(frame["word_end"]-frame["word_start"]) # 2単語目以降は I-FrameID形式
    return target,tags


def encode_opencalm(data,params):
    input = []
    target = []
    cls = []
    sep = []
    args_list = []
    frame_list = []
    frame_ = []
    if params["cls"]:
        cls = ["<|endoftext|>"]
    if params["sep"]:
        sep = ["<|endoftext|>"]

    for i,d in enumerate(data):
        #input.append(cls+d["text"].split()+sep)
        # try:
        #     for arg in d["args"]:
        #         if arg["word_start"] == -1:
        #             continue
        # except KeyError:
        #     #print(data)
        #     print(i)
        #     exit()
        input.append(cls+d["sentence"].split()+sep)
        if params["bio"]:
            target_seq,arg_tags = make_bioseq_opencalm(d,len(cls),len(sep),params["arg_role"],params["not_O_fidtag"]) 
            target.append(target_seq)
            for arg in arg_tags:
                if not arg in args_list:
                    args_list.append(arg)
        # else:
        #     target.append(make_target_seq(d,len(cls),len(sep),params["arg_role"]))
        # frame = int(d["predicate"]["frameID"].split(":")[0])
        frame = d["predicate"]["frameID"]
        if params["not_O_fidtag"] and (not params["predicate_is_BP"]):
            if not frame in frame_list:
                frame_list.append(frame)
    
    args_list.sort()
    frame_list.sort()
    
    return input,target,args_list+frame_list

def encode_fidin_opencalm(data,params):
    input = []
    target = []
    cls = []
    sep = []
    args_list = []
    frame_list = []
    frame_ = []
    if params["cls"]:
        cls = ["<|endoftext|>"]
    if params["sep"]:
        sep = ["<|endoftext|>"]

    for i,d in enumerate(data):
        #input.append(cls+d["text"].split()+sep)
        # try:
        #     for arg in d["args"]:
        #         if arg["word_start"] == -1:
        #             continue
        # except KeyError:
        #     #print(data)
        #     print(i)
        #     exit()
        input.append(cls+d["sentence"].split()+sep)
        if params["bio"]:
            target_seq,arg_tags = make_bioseq_opencalm(d,len(cls),len(sep),params["arg_role"],params["not_O_fidtag"])
            target.append(target_seq)
            for arg in arg_tags:
                if not arg in args_list:
                    args_list.append(arg)
        # else:
        #     target.append(make_target_seq(d,len(cls),len(sep),params["arg_role"]))
        # frame = int(d["predicate"]["frameID"].split(":")[0])
        frame = d["predicate"]["frameID"]
        if params["not_O_fidtag"] and (not params["predicate_is_BP"]): # O(オー)でもBPでもない B-FIDのとき
            if not frame in frame_list:
                frame_list.append(frame)
    
    args_list.sort()
    frame_list.sort()
    
    return input,target,args_list,frame_list

def syntax_encode_opencalm(syntax_json,use_parents_node,use_child_node): # 自己ループはtorchのGCNConvのパラメータで設定 use_parents_node use_child_nodeが両方ともTrueで無向(双方向有向)グラフ
    tag2id = {}
    id_count = 0
    new_syntax_dic = {}
    for sentenceid,list_ in syntax_json.items(): # それぞれ文章の構文構造について
        from_edge = []
        to_edge = []
        tag_list = []
        position_list = []
        for i,json_ in enumerate(list_): # 構文構造内の各構文タグでループ
            if not json_["tag"] in tag2id: # 構文タグの辞書を作成
                tag2id[json_["tag"]] = id_count
                id_count += 1

            tag_list.append(tag2id[json_["tag"]]) # 構文タグのIDでリスト作成
            position_list.append([json_["word_start"],json_["word_end"]]) # タグの単語範囲のリスト作成

            if use_parents_node and not json_["parent_index"] == -1: # rootノードの親ノードはないから -1 で設定している
                from_edge.append(i)  # 自身のノードid
                to_edge.append(json_["parent_index"]) # 親ノード
            if use_child_node:
                from_edge.extend([i]*len(json_["child_index"])) # 自身のノードidを子ノード分作成
                to_edge.extend(json_["child_index"]) # 子ノードのID
            # [[1, 0, 0, 0]
            #  [0, 2, 3, 4]] みたいな形式にするために作成 
            # この例の時, 0の親ノードは 1 で0の子ノードは 2,3,4 上下でノードが繋がってる
        new_syntax_dic[sentenceid] = {"tag":tag_list,"word_position": position_list,"edge_rep":[from_edge,to_edge]} # 取り出しやすくするために辞書形式で代入

    id2tag = {value: key for key, value in tag2id.items()} # 構文タグのid形式でとタグの対応関係辞書を作成
    return new_syntax_dic,id2tag


def encode_multi_opencalm(data,params):
    input = []
    fid_targets = []
    bio_targets = []
    target_candidate = []
    cls = []
    sep = []
    args_list = []
    frame_list = []
    frame_ = []
    if params["cls"]: # 先頭にclsをつける
        cls = ["<|endoftext|>"]
    if params["sep"]: # 末尾にsepをつける
        sep = ["<|endoftext|>"]

    for i,d in enumerate(data): # 全てのデータセットについて学習に必要なものを取り出す
        if params["with_predicate"]: # True  # 文章のあとに述語を連結する cls 文章 sep 述語 sep　形式
            input.append(cls + d["sentence"].split() + sep + d["sentence"].split()[d["predicate"]["word_start"]:d["predicate"]["word_end"]+1] + sep) # [d["predicate"]["word_start"]:d["predicate"]["word_end"]+1] で述語位置の単語を取得
            fid_targets.append(d["predicate"]["frameID"]) # FrameIDの正解を取得
            target_candidate.append(d["predicate"]["candidate"]) # 正解候補を取得
        else: # cls 文章 sep の形式
            input.append(cls + d["sentence"].split() + sep)
        if params["bio"]: # True 
            target_seq,arg_tags = make_bioseq_opencalm(d,len(cls),len(sep),params["arg_role"],params["not_O_fidtag"]) # BIO形式の出力を作成
            bio_targets.append(target_seq)
            for arg in arg_tags:
                if not arg in args_list:
                    args_list.append(arg) # 重複無しで意味役割の一覧リスト作成

        if params["setFID"]: # Falseだからスキップ  FrameIDのリスト作成
            frame = d["predicate"]["frameID"]
            if not frame in frame_list:
                frame_list.append(frame)
    
    # print(input[0:5])


    args_list.sort() # 見やすくするためにソート
    frame_list.sort()
    
    return input,bio_targets, fid_targets,args_list,frame_list, target_candidate