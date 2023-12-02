import torch
import torch.nn.utils.rnn as rnn
import torch.nn.functional as functional

def seq_padding(data):
    seq = [torch.LongTensor(d[0]) for d in data]
    #same = [d[1] for d in data]
    #max_ = max(same)
    #max_ = 31
    #same = torch.LongTensor(same)
    pad = rnn.pad_sequence(seq,batch_first=True,padding_value=0)
    #print(pad.shape)
    #one_hot = functional.one_hot(same,num_classes=max_+1)
    #print(one_hot.shape)
    return pad#,one_hot

def seq_padding_v2(data):
    seq = [torch.LongTensor(d) for d in data]
    pad = rnn.pad_sequence(seq,batch_first=True,padding_value=0)

    return pad

def seq_padding_v3(seq, padding_size): # 処理結果は↑と一緒
    target = torch.zeros(len(seq),padding_size,dtype=torch.long)
    for i,s in enumerate(seq):
        target[i][:len(s[0])] = torch.LongTensor(s[0])
    return target

# def srl_matrix(data,srl2id,seq_length,first_frameid,params,test=False):
#     data_legth = len(data)
#     srl_length = len(srl2id)
#     row_len = 1
#     start_end = {"0,0":0}
#     max_sa = [0]
#     #print(seq_length)
#     # decoder でも同じ辞書を作成している
#     for i in range(1,seq_length-1): #末尾の</s>を考慮  
#         for j in range(1,seq_length-1):
#             if i > j:
#                 continue
#             start_end["{},{}".format(i,j)] = row_len
#             row_len += 1
#     #matrix = torch.zeros(params['batch_size'],row_len,srl_length,dtype=torch.float)
#     if not test:
#         #matrix = torch.zeros(params['batch_size'],row_len,dtype=torch.long)
#         for i,seq_srl in enumerate(data):
#             for start,end,srl in seq_srl:
#                 if srl >=first_frameid:
#                     key = srl
#                     break
#             if i == 0:
#                 matrix = torch.full((1,row_len),fill_value=key)
#             else:
#                 matrix = torch.cat((matrix,torch.full((1,row_len),fill_value=key)),dim=0)

#     else:
#         #matrix = torch.zeros(1,row_len,dtype=torch.long)
#         for i,seq_srl in enumerate(data):
#             for start,end,srl in seq_srl:
#                 if srl >=first_frameid:
#                     key = srl
#                     break
#         matrix = torch.full((1,row_len),fill_value=key)
#     for i,seq_srl in enumerate(data):
#         for start,end,srl in seq_srl:
#             if srl < first_frameid:
#                 num = start_end["{},{}".format(start,end)]
#                 #matrix[i][num][srl] = 1
#                 matrix[i][num] = srl
#     return matrix,start_end

# def dummy_srl_matrix(data,srl2id,seq_length,first_frameid,frame_se,params,test=False):
#     row_len = 1
#     start_end = {"0,0":0}
#     matrix = None
#     for i in range(1,seq_length-1): #末尾の</s>を考慮  
#         for j in range(1,seq_length-1):
#             if i > j:
#                 continue
#             start_end["{},{}".format(i,j)] = row_len
#             row_len += 1
#     #matrix = torch.zeros(params['batch_size'],row_len,srl_length,dtype=torch.float)
#     if not test:
#         #matrix = torch.zeros(params['batch_size'],first_frameid,dtype=torch.long)
#         for s,e in frame_se:
#             if matrix == None:
#                 matrix = torch.full((1,first_frameid-1),fill_value=start_end["{},{}".format(s,e)])
#             else:
#                 matrix = torch.cat((matrix,torch.full((1,first_frameid-1),fill_value=start_end["{},{}".format(s,e)])),dim=0)
#     else:
#         #matrix = torch.zeros(1,first_frameid,dtype=torch.long)
#         matrix = torch.full((1,first_frameid-1),fill_value=start_end["{},{}".format(frame_se[0][0],frame_se[0][1])])
#     for i,seq_srl in enumerate(data):
#         for start,end,srl in seq_srl:
#             if srl < first_frameid and srl > 0:
#                 num = start_end["{},{}".format(start,end)]
#                 #matrix[i][num][srl] = 1
#                 matrix[i][srl-1] = num
#     return matrix,start_end

# def reshape_onehot(x_onehot,length):
#     new_onehot = None
#     for a_onehot in x_onehot:
#         a_onehot = torch.unsqueeze(a_onehot,1)
#         a_onehot = a_onehot.repeat(1,length)
#         a_onehot = torch.t(a_onehot)
#         a_onehot = torch.unsqueeze(a_onehot,0)
#         if new_onehot == None:
#             new_onehot = a_onehot
#         else:
#             new_onehot = torch.cat((new_onehot,a_onehot),dim=0)
#     return new_onehot

# def frame_point(target,padding,first_frameid):
#     frame_vec = None
#     se_list = []
#     for t in target:
#         frame_flag = True
#         for start,end,srl in t:
#             if srl >= first_frameid and frame_flag:
#                 frame = torch.zeros(padding,dtype=torch.long)
#                 frame[start:end+1] = 1
#                 frame = torch.unsqueeze(frame,0)
#                 frame_flag = False
#                 #print(frame.shape)
#                 #print(frame)
#                 se_list.append([start,end])
#                 if frame_vec == None:
#                     frame_vec = frame
#                 else:
#                     frame_vec = torch.cat((frame_vec,frame),dim=0)
#     frame_vec = torch.unsqueeze(frame_vec,2)
#     #print(frame_vec.shape)
#     return frame_vec,se_list

# def frame_target(target,frame_id):
#     frame = torch.zeros(len(target),dtype=torch.long)
#     frame_point = []
#     for i,(s,e,srl) in enumerate(target):
#         if srl in frame_id:
#             frame[i] = frame_id.index(srl)
#             frame_point.append([s,e])
#     return frame,frame_point


##################################################
# def make_frameposition(frame_point,padding):
#     frame_vec = None
#     for _,index in frame_point:
#         frame = torch.zeros(padding,dtype=torch.long)
#         frame[index] = 1
#         frame = torch.unsqueeze(frame,0)
#         if frame_vec == None:
#             frame_vec = frame
#         else:
#             frame_vec = torch.cat((frame_vec,frame),dim=0)
#     frame_vec = torch.unsqueeze(frame_vec,2)
#     return frame_vec

# def make_mask(x_seq,batchsize,padding,test=False): 
#     mask_vec = None
#     if test: # テストは1個だけ 作成
#         mask_vec = torch.ones(padding,dtype=torch.uint8)
#         mask_vec = torch.unsqueeze(mask_vec,0)
#     else: 
#         for i in range(len(x_seq)): # 1文章分ずつmaskを作る (バッチサイズ分,一気に作成すればよかった)
#             mask = torch.zeros(padding,dtype=torch.uint8) 
#             length = len(x_seq[i])
#             mask[:length] = 1 # 文章部分を1にする
#             mask = torch.unsqueeze(mask,0)
#             if mask_vec == None:
#                 mask_vec = mask
#             else:
#                 mask_vec = torch.cat((mask_vec,mask),dim=0)
#     return mask_vec

# def make_frameposition_fidin(frame_point,padding,frameid):
#     frame_vec = None
#     for _,index,_ in frame_point:  # 3つ目追加するかも
#         frame = torch.zeros(padding,dtype=torch.long)
#         #if frameid:
#         #    for i,token in enumerate(target):
#         #        if token[2:].isdecimal():
#         #            index = i
#         #else:
#         #    index = target.index(-1)
#         frame[index] = 1
#         frame = torch.unsqueeze(frame,0)
#         if frame_vec == None:
#             frame_vec = frame
#         else:
#             frame_vec = torch.cat((frame_vec,frame),dim=0)
#     frame_vec = torch.unsqueeze(frame_vec,2)
#     return frame_vec

# def make_frameposition_and_id(frame_point,padding,frame_list):
#     frame_vec = None
#     frameid_counts = len(frame_list)
#     for i,(_,index,id_index) in enumerate(frame_point):
#         frame_position = torch.zeros(padding,dtype=torch.long)
#         frame_position[index] = 1

#         frame_id = torch.zeros(frameid_counts,dtype=torch.long)
#         frame_id[id_index] = 1
#         frame_id = frame_id.expand(padding,frameid_counts) # expandはメモリの保存場所は同一だから後の操作に注意

#         frame_position = torch.unsqueeze(frame_position,0)
#         frame_position = torch.unsqueeze(frame_position,2)
#         frame_id = torch.unsqueeze(frame_id,0)

#         frame = torch.cat((frame_position,frame_id),dim=2)
#         if frame_vec == None:
#             frame_vec = frame
#         else:
#             frame_vec = torch.cat((frame_vec,frame),dim=0)
#     return frame_vec

### model is OpenCALM ###
### 文章が<|endoftext|>,私,は,歩く,<|endoftext|>,歩く,<|endoftext|>のようになっている
def make_token_type_opencalm(x, batch_size, max_len): # 2番目の<|endoftext|>の次から 末尾の<|endoftext|>まで1 
    token_type = torch.zeros(batch_size,max_len, dtype=torch.long)
    for i,seq in enumerate(x):
        second_eot_index = seq.index(0, seq.index(0) + 1) # 2番目の<|endoftext|>のindexを取得 <|endoftext|> == 0
        seq_length = len(seq) # 文章長を取得
        token_type[i][second_eot_index+1:seq_length] = 1 # 最初の<|endoftext|>の次の位置から文章の末尾まで1にする
    return token_type

def make_predicate_position_opencalm(x,batch_size,max_len): # 末尾のSEPを除く
    position = torch.zeros(batch_size,max_len,1,dtype=torch.uint8)
    for i,seq in enumerate(x):
        second_eot_index = seq.index(0, seq.index(0) + 1)
        position[i,second_eot_index+1:len(seq)-1,0] = 1
    return position

# def make_predicate_position_v2(x,batch_size,max_len):
#     position = torch.zeros(batch_size,max_len,1,dtype=torch.bool)
#     for i,seq in enumerate(x):
#         position[i][0][0] = 1
#         position[i][len(seq)-2][0] = 1
#     return position

# def fid_candidate(target_candidate,batch_size,target_length):
#     candidate = torch.zeros(batch_size,target_length,dtype=torch.uint8) # batch * FrameIDの種類数(1096次元)
#     for i,candidate_list in enumerate(target_candidate):
#         if candidate_list == [-1]: # 候補が同定できていない
#             candidate[i][:] = 1 # すべての次元が1
#         else:
#             candidate[i][candidate_list] = 1 # 候補次元のみ1
#     return candidate

# def make_output_mask(target_candidate,batch_size,target_length): # 上のfid_candidateと一緒 dtypeの違いだけ
#     mask = torch.zeros(batch_size,target_length,dtype=torch.int8) 
#     for i,candidate in enumerate(target_candidate):
#         if candidate == -1:
#             mask[i][:] = 1
#         else:
#             mask[i][candidate] = 1
#     return mask
##################################################





def make_frameposition(frame_point,padding):
    frame_vec = None
    for _,index in frame_point:
        frame = torch.zeros(padding,dtype=torch.long)
        frame[index] = 1
        frame = torch.unsqueeze(frame,0)
        if frame_vec == None:
            frame_vec = frame
        else:
            frame_vec = torch.cat((frame_vec,frame),dim=0)
    frame_vec = torch.unsqueeze(frame_vec,2)
    return frame_vec

def make_mask(x_seq,batchsize,padding,test=False): 
    mask_vec = None
    if test: # テストは1個だけ 作成
        mask_vec = torch.ones(padding,dtype=torch.uint8)
        mask_vec = torch.unsqueeze(mask_vec,0)
    else: 
        for i in range(len(x_seq)): # 1文章分ずつmaskを作る (バッチサイズ分,一気に作成すればよかった)
            mask = torch.zeros(padding,dtype=torch.uint8) 
            length = len(x_seq[i])
            mask[:length] = 1 # 文章部分を1にする
            mask = torch.unsqueeze(mask,0)
            if mask_vec == None:
                mask_vec = mask
            else:
                mask_vec = torch.cat((mask_vec,mask),dim=0)
    return mask_vec

def make_frameposition_fidin(frame_point,padding,frameid):
    frame_vec = None
    for _,index,_ in frame_point:  # 3つ目追加するかも
        frame = torch.zeros(padding,dtype=torch.long)
        #if frameid:
        #    for i,token in enumerate(target):
        #        if token[2:].isdecimal():
        #            index = i
        #else:
        #    index = target.index(-1)
        frame[index] = 1
        frame = torch.unsqueeze(frame,0)
        if frame_vec == None:
            frame_vec = frame
        else:
            frame_vec = torch.cat((frame_vec,frame),dim=0)
    frame_vec = torch.unsqueeze(frame_vec,2)
    return frame_vec

def make_frameposition_and_id(frame_point,padding,frame_list):
    frame_vec = None
    frameid_counts = len(frame_list)
    for i,(_,index,id_index) in enumerate(frame_point):
        frame_position = torch.zeros(padding,dtype=torch.long)
        frame_position[index] = 1

        frame_id = torch.zeros(frameid_counts,dtype=torch.long)
        frame_id[id_index] = 1
        frame_id = frame_id.expand(padding,frameid_counts) # expandはメモリの保存場所は同一だから後の操作に注意

        frame_position = torch.unsqueeze(frame_position,0)
        frame_position = torch.unsqueeze(frame_position,2)
        frame_id = torch.unsqueeze(frame_id,0)

        frame = torch.cat((frame_position,frame_id),dim=2)
        if frame_vec == None:
            frame_vec = frame
        else:
            frame_vec = torch.cat((frame_vec,frame),dim=0)
    return frame_vec

### 述語の部分に1を立てている
def make_token_type(x, batch_size, max_len): # 最初のSEPの次から 末尾のSEPまで1 
    token_type = torch.zeros(batch_size,max_len, dtype=torch.long)
    for i,seq in enumerate(x):        
        # print("before token_type is ",token_type[i])
        sep_index = seq.index(3) # 最初の[SEP]の位置を取得
        # print(" sep_index",sep_index)
        seq_length = len(seq) # 文章長を取得
        # print("seq_length",seq_length)
        token_type[i][sep_index+1:seq_length] = 1 # 最初の[SEP]の次の位置から文章の末尾まで1にする
        # print(" after token_type is ",token_type[i])
        # exit(0)
    return token_type

def make_predicate_position(x,batch_size,max_len): # 末尾のSEPを除く
    position = torch.zeros(batch_size,max_len,1,dtype=torch.uint8)
    # print("before position",position[0])
    for i,seq in enumerate(x):
        # print("seq",seq)
        position[i,seq.index(3)+1:len(seq)-1,0] = 1
        # print(" after position",position[i])
    return position

def make_predicate_position_v2(x,batch_size,max_len):
    position = torch.zeros(batch_size,max_len,1,dtype=torch.bool)
    for i,seq in enumerate(x):
        position[i][0][0] = 1
        position[i][len(seq)-2][0] = 1
    return position

def fid_candidate(target_candidate,batch_size,target_length):
    candidate = torch.zeros(batch_size,target_length,dtype=torch.uint8) # batch * FrameIDの種類数(1096次元)
    for i,candidate_list in enumerate(target_candidate):
        if candidate_list == [-1]: # 候補が同定できていない
            candidate[i][:] = 1 # すべての次元が1
        else:
            candidate[i][candidate_list] = 1 # 候補次元のみ1
    return candidate

def make_output_mask(target_candidate,batch_size,target_length): # 上のfid_candidateと一緒 dtypeの違いだけ
    mask = torch.zeros(batch_size,target_length,dtype=torch.int8) 
    for i,candidate in enumerate(target_candidate):
        if candidate == -1:
            mask[i][:] = 1
        else:
            mask[i][candidate] = 1
    return mask
