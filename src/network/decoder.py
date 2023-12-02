from itertools import count
import numpy as np
import random
from transformers import BertJapaneseTokenizer
from numpy.core.fromnumeric import argsort
from sklearn.model_selection import train_test_split

# def calculate(pred,true,firstid,id2srl,id_list):
#     count = 0
#     frame = 0
#     id = 0
#     true_list = []
#     pred_list = []
#     srl_count = {}   # srl: [true,pred,match]
#     for t in true:
#         srl = id2srl[t[2]]
#         if t[0] == 0:
#             split = srl.split("arg")
#             #print(srl)
#             if len(split) != 1:
#                 for s in split[1:]:
#                     arg = "arg{}".format(s)
#                     true_list.append([0,0,arg])
#                     if not arg in srl_count:
#                         srl_count[arg] = [1,0,0]
#                     else:
#                         srl_count[arg][0] += 1
#                 continue
#         true_list.append([t[0],t[1],srl])
#         if "arg" in srl:
#             if not srl in srl_count:
#                 srl_count[srl] = [1,0,0]
#             else:
#                 srl_count[srl][0] += 1
#         if srl.isdecimal():
#             fs = t[0]
#             fe = t[1]
#             frameid = srl
#     #print(pred)
#     for p in pred:
#         srl = p[2]
#         if p[0] == 0:
#             split = srl.split("arg")
#             if len(split) != 1:
#                 for s in split[1:]:
#                     arg = "arg{}".format(s)
#                     pred_list.append([0,0,arg])
#                     if not arg in srl_count:
#                         srl_count[arg] = [0,1,0]
#                     else:
#                         srl_count[arg][1] += 1
#                 continue
#         pred_list.append([p[0],p[1],srl])
#         if "arg" in srl:
#             if not srl in srl_count:
#                 srl_count[srl] = [0,1,0]
#             else:
#                 srl_count[srl][1] += 1
#     #print("true:",true_list)
#     #print("pred:",pred_list)

#     for p in pred_list:
#         if "arg" in p[2] and p in true_list:
#             count += 1
#             srl_count[p[2]][2] += 1
#         if p[0] == fs and p[1] == fe:
#             if p[2].isdecimal():
#                 if p[2] == frameid:
#                     frame = 1
#                 id = frameid
#             else:
#                 if id_list != []:
#                     id = random.choice(id_list)
#                     if id == frameid:
#                         frame = 1
#                 else:
#                     id = random.randrange(1,1005)
#     #print(srl_count)
    
#     if id_list == []: #dummy用
#         if len(pred_list) >= 1:
#             precision = count / (len(pred_list)) #概念フレームの個数分を引く
#             recall = count / (len(true_list)-1)
#         else:
#             precision = 0
#             recall = 0
#     else:
#         if len(pred_list) > 1:
#             precision = count / (len(pred_list)-1) #概念フレームの個数分を引く
#             recall = count / (len(true_list)-1)
#         else:
#             precision = 0
#             recall = 0
#     #print(count)
#     #print(precision,recall)
#     # print(frame)
#     # print()
    
#     if precision == 0 and recall == 0:
#         f1 = 0
#     else:
#         f1 = 2*((precision*recall) / (precision+recall))
#     return precision,recall,f1,frame,srl_count,id


# def decode(x_test,t_test,all_preds,all_start_end,id2word,id2srl,first_frameid,frame_dic,base_dic,frame_pred,mast_frame):
#     output_data = []
#     #prf_score = [0.0,0.0,0.0]  # precision, recall ,f1の順
#     true_frame = 0.
#     #analy = {0:{"f1":0.0,"count":0,"frame":0},1:{"f1":0.0,"count":0,"frame":0},2:{"f1":0.0,"count":0,"frame":0},3:{"f1":0.0,"count":0,"frame":0},4:{"f1":0.0,"count":0,"frame":0},5:{"f1":0.0,"count":0,"frame":0},6:{"f1":0.0,"count":0,"frame":0},7:{"f1":0.0,"count":0,"frame":0},8:{"f1":0.0,"count":0,"frame":0},9:{"f1":0.0,"count":0,"frame":0},10:{"f1":0.0,"count":0,"frame":0},11:{"f1":0.0,"count":0,"frame":0},12:{"f1":0.0,"count":0,"frame":0},13:{"f1":0.0,"count":0,"frame":0},14:{"f1":0.0,"count":0,"frame":0},15:{"f1":0.0,"count":0,"frame":0},16:{"f1":0.0,"count":0,"frame":0},17:{"f1":0.0,"count":0,"frame":0}}
#     srl_analy = {}
#     for i,start_end in enumerate(all_start_end):
#         word_flag = [0]*len(x_test[i][0])
#         word_flag[-1] = 1
#         pred_ss = []
#         arg_max = np.ravel(np.argmax(all_preds[i], axis=1))   # 各スパンに割り振られたsrl
#         score_max = np.ravel(np.amax(all_preds[i], axis=1))   # 各スパンの最大値スコア
#         arg_sort = np.ravel(np.argsort(-score_max))   # 0からスコアの最大値のインデックス
#         idx = 0


#         t_test[i].sort(key=lambda x:x[0])
#         true_idx = 0
#         for s,e,t in t_test[i]:
#             if s == 0 and e == 0:
#                 true_idx += 1
#             if t >= first_frameid:
#                 fs = s
#                 fe = e
#         if mast_frame:
#             span_val = start_end["{},{}".format(fs,fe)]
#             word_flag[fs:fe+1] = [1]*(fe+1-fs)
#             #srl = arg_max[span_val]
#             #pred_ss.append([fs,fe,id2srl[srl]])
#         #print(t_test[i])
#         #print(pred_ss)

#         start_end = {v: k for k,v in start_end.items()} #keyとvalの入れ替え

#         #print("argmax:",arg_max.shape)
#         #print("score_max:",score_max.shape)
#         #print("arg_sort:",arg_sort.shape)
#         while 0 in word_flag:
#             #print(word_flag)
#             span_val = arg_sort[idx]
#             span = start_end[span_val].split(",")
#             #print(*span)
#             span = [int(s) for s in span]
#             if not 1 in word_flag[span[0]:span[1]+1]:
#                 word_flag[span[0]:span[1]+1] = [1]*(span[1]+1-span[0])
#                 srl = arg_max[span_val]
#                 if srl != 0 and srl < first_frameid:
#                     pred_ss.append([span[0],span[1],id2srl[srl]])
#             idx += 1
#         pred_ss.sort(key=lambda x:x[0])
#         #print("pre:",pred_ss)
#         #print("tru:",t_test[i])
#         start_idx = [s for s,e,t in t_test[i] if t >= first_frameid]
#         #if start_idx[0] != 0:
#         #    frame_word = id2word[x_test[i][0][start_idx[0]]]
#         #    id_list = frame_dic[base_dic[frame_word]]
#         #else:
#         id_list = []
#         pre,rec,f1,a_frame,srl_count,pre_srl = calculate(pred_ss,t_test[i],first_frameid,id2srl,id_list)
        
#         for j,(s,_,_) in enumerate(pred_ss):
#             if s == start_idx[0]:
#                 pred_ss[j][2] = pre_srl
        
#         prf_score[0] += pre
#         prf_score[1] += rec
#         prf_score[2] += f1
#         true_frame += a_frame

#         key = int(len(x_test[i][0])/10)
#         #analy[key]["f1"] += f1
#         #analy[key]["count"] += 1
#         #analy[key]["frame"] += a_frame

#         for k,val in srl_count.items():
#             if k in srl_analy:
#                 srl_analy[k]["true"] += val[0]
#                 srl_analy[k]["pre"] += val[1]
#                 srl_analy[k]["match"] += val[2]
#             else:
#                 srl_analy[k] = {"true":val[0],"pre":val[1],"match":val[2]}
        
#         if true_idx != 0:
#             true_idx -= 1
#         pred_list = []
#         true_list = []
#         pred_idx = 0
#         pred_flag = False
#         true_flag = False
#         for j,word in enumerate(x_test[i][0]):
#             try:
#                 if pred_flag == True and j == pred_ss[pred_idx][1]+1:
#                     if pred_ss[pred_idx][2] != 0:
#                         pred_list.append("</{}>".format(pred_ss[pred_idx][2]))
#                     pred_flag = False
#                     pred_idx += 1
#                 if true_flag == True and j == t_test[i][true_idx][1]+1:
#                     label = id2srl[t_test[i][true_idx][2]]
#                     true_list.append("</{}>".format(label))
#                     if not "arg" in label:
#                         if frame_pred[i] == -1:
#                             pred_list.append("</{}>".format(label))
#                         else:
#                             pred_list.append("</{}>".format(frame_pred[i]))
#                     true_flag = False
#                     true_idx += 1

#                 if pred_flag == False and pred_idx < len(pred_ss) and j == pred_ss[pred_idx][0]:
#                     if pred_ss[pred_idx][2] != 0:
#                         pred_list.append("<{}>".format(pred_ss[pred_idx][2]))
#                     pred_flag = True
#                 if true_flag == False and true_idx < len(t_test[i]) and j == t_test[i][true_idx][0]:
#                     label = id2srl[t_test[i][true_idx][2]]
#                     true_list.append("<{}>".format(label))
#                     if not "arg" in label:
#                         if frame_pred[i] == -1:
#                             pred_list.append("<{}>".format(label))
#                         else:
#                             pred_list.append("<{}>".format(frame_pred[i]))
#                     true_flag = True
#                 pred_list.append(id2word[word])
#                 true_list.append(id2word[word])
#             except IndexError:
#                 print("true idx:",true_idx)
#                 print("pred idx:",pred_idx)
#                 print(len(pred_ss))
#                 print(i)
#                 exit()

#         #print(*pred_list)
#         #print(*true_list)
#         output_data.append([true_list,pred_list])

#     #print(prf_score)
#     #print(true_frame)
#     pre = prf_score[0] / len(t_test)
#     rec = prf_score[1] / len(t_test)
#     f1 = prf_score[2] / len(t_test)
#     frame_acc = true_frame / len(t_test)

#     print("all precision score:",pre)
#     print("all recall score",rec)
#     print("all f1 score:",f1)
#     print("all frame acc:",frame_acc)

#     with open("./../data/output/score.txt",mode="w") as f:
#         print("all precision score:",pre,file=f)
#         print("all recall score",rec,file=f)
#         print("all f1 score:",f1,file=f)
#         print("all frame acc:",frame_acc,file=f)
#         print("",file=f)

#         for key,value in analy.items():
#             if value["count"] == 0:
#                 continue
#             print("f1: {} < seq len < {}: {:.4}".format(key*10,(key+1)*10,value["f1"]/value["count"]))
#             print("f1: {} < seq len < {}: {:.4}".format(key*10,(key+1)*10,value["f1"]/value["count"]),file=f)
#         print("",file=f)

#         for key,value in analy.items():
#             print(key*10,value,file=f)


#     with open("./../data/output/score.txt",mode="a") as f:
#         print("",file=f)
#         for key,value in srl_analy.items():
#             if value["pre"] != 0:
#                 pre = value["match"]/value["pre"]
#             else:
#                 pre = 0.0
#             if value["true"] != 0:
#                 rec = value["match"]/value["true"]
#             else:
#                 rec = 0.0
#             if pre == 0 and rec == 0:
#                 f1 = 0.0
#             else:
#                 f1 = 2*((pre*rec) / (pre+rec))
#             print("{}  precision: {:.4} recall: {:.4} f1: {:.4}".format(key,pre,rec,f1))
#             print("{}  precision: {:.4} recall: {:.4} f1: {:.4}".format(key,pre,rec,f1),file=f)
    
#     with open("./../data/output/score.txt",mode="a") as f:
#         print("",file=f)
#         for key,val in srl_analy.items():
#             print("{} {}".format(key,val),file=f)
#     #print("exit")
#     #exit()
#     return output_data

# def dummy_decode(x_test,t_test,all_preds,all_start_end,id2word,id2srl,first_frameid,frame_dic,base_dic,frame_preds,mast_frame):
#     output_data = []
#     prf_score = [0.0,0.0,0.0]  # precision, recall ,f1の順
#     true_frame = 0
#     count_ = 0
#     fcount = 0
#     for i,start_end in enumerate(all_start_end):
#         arg_max = np.ravel(np.argmax(all_preds[i], axis=1))   # 各スパンに割り振られたsrl
#         score_max = np.ravel(np.amax(all_preds[i], axis=1))   # 各スパンの最大値スコア
#         arg_sort = np.ravel(np.argsort(-score_max))   # 0からスコアの最大値のインデックス
#         # print(arg_max)
#         # print(score_max)
#         # print(arg_sort)
#         word_flag = [0]*len(x_test[i][0])
#         pred_ss = []
#         for s,e,sr in t_test[i]:
#             if sr >= first_frameid:
#                 frame_span = start_end["{},{}".format(s,e)]
#                 if frame_preds[i] == -1 or id2srl[sr] == frame_preds[i]:
#                     true_frame += 1
#                 if frame_preds[i] != -1:
#                     count_ += 1
#                 if id2srl[sr] == frame_preds[i]:
#                     #print(frame_preds[i])
#                     fcount += 1
#         start_end = {v: k for k,v in start_end.items()}
#         for arg in arg_sort:
#             #arg = arg + 1
#             #if arg == 0:
#             #    continue
#             spanid = arg_max[arg]
#             #print(arg,spanid,frame_span)
#             if spanid == frame_span:
#                 continue
#             span = start_end[spanid].split(",")
#             #print(span)
#             span = [int(s) for s in span]
#             if not 1 in word_flag[span[0]:span[1]+1]:
#                 word_flag[span[0]:span[1]+1] = [1]*(span[1]+1-span[0])
#                 pred_ss.append([span[0],span[1],id2srl[arg+1]])
#             else:
#                 tmp_as = np.squeeze(all_preds[i])
#                 tmp_as = tmp_as[:,arg]

#                 tmp_asort = np.ravel(np.argsort(-tmp_as))
#                 for tmp in tmp_asort[1:]:
#                     if tmp == frame_span:
#                         break
#                     start_end
#                     span = start_end[tmp].split(",")
#                     span = [int(s) for s in span]
#                     if not 1 in word_flag[span[0]:span[1]+1]:
#                         word_flag[span[0]:span[1]+1] = [1]*(span[1]+1-span[0])
#                         pred_ss.append([span[0],span[1],id2srl[arg+1]])
#                         break
#         #print(pred_ss)
#         pred_ss.sort(key=lambda x:x[0])
#         pre,rec,f1,a_frame,srl_count,pre_srl = calculate(pred_ss,t_test[i],first_frameid,id2srl,[])
        

#         prf_score[0] += pre
#         prf_score[1] += rec
#         prf_score[2] += f1

#         pred_list = []
#         true_list = []
#         pred_idx = 0
#         true_idx = 0
#         pred_flag = False
#         true_flag = False
#         # print(x_test[i][0])
#         # print(t_test[i])
#         # print(pred_ss)
#         for j,word in enumerate(x_test[i][0]):
#             try:
#                 if pred_flag == True and j == pred_ss[pred_idx][1]+1:
#                     if pred_ss[pred_idx][2] != 0:
#                         pred_list.append("</{}>".format(pred_ss[pred_idx][2]))
#                     pred_flag = False
#                     pred_idx += 1
#                 if true_flag == True and j == t_test[i][true_idx][1]+1:
#                     label = id2srl[t_test[i][true_idx][2]]
#                     true_list.append("</{}>".format(label))
#                     if not "arg" in label:
#                         if frame_preds[i] == -1:
#                             pred_list.append("</{}>".format(label))
#                         else:
#                             pred_list.append("</{}>".format(frame_preds[i]))
#                     true_flag = False
#                     true_idx += 1

#                 if pred_flag == False and pred_idx < len(pred_ss) and j == pred_ss[pred_idx][0]:
#                     if pred_ss[pred_idx][2] != 0:
#                         pred_list.append("<{}>".format(pred_ss[pred_idx][2]))
#                     pred_flag = True
#                 if true_flag == False and true_idx < len(t_test[i]) and j == t_test[i][true_idx][0]:
#                     label = id2srl[t_test[i][true_idx][2]]
#                     true_list.append("<{}>".format(label))
#                     if not "arg" in label:
#                         if frame_preds[i] == -1:
#                             pred_list.append("<{}>".format(label))
#                         else:
#                             pred_list.append("<{}>".format(frame_preds[i]))
#                     true_flag = True
#                 pred_list.append(id2word[word])
#                 true_list.append(id2word[word])
#             except IndexError:
#                 print(*true_list)
#                 print(*pred_list)
#                 print("true idx:",true_idx)
#                 print("pred idx:",pred_idx)
#                 print(len(pred_ss))
#                 print(i)
#                 print(j)
#                 exit()
#         #print(*true_list)
#         #print(*pred_list)
#         output_data.append([true_list,pred_list])
#         #print()
        

#     #print(prf_score)
#     #print(true_frame)
#     pre = prf_score[0] / len(t_test)
#     rec = prf_score[1] / len(t_test)
#     f1 = prf_score[2] / len(t_test)

#     print("all precision score:",pre)
#     print("all recall score",rec)
#     print("all f1 score:",f1)
#     print("true,frame",true_frame/len(t_test))
#     print("count",count_)
#     print("fcount",fcount)
#     print("fc/c",fcount/count_)

#     return output_data

# ------------------ BIO開発後 ------------------

def chenge_id2bio(preds,id2bio):
    new_preds = []
    for pred in preds:
        if len(pred) == 2: # 常にFalseになる (昔の残骸)
            pred = pred[0]
        seq = []
        for token in pred:
            seq.append(id2bio[token]) # 意味役割のID系列をBIO系列に変換
        new_preds.append(seq)
    return new_preds

# def change_bertid2seq(seqs):
#     new_seqs = []
#     tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
#     for seq in seqs:
#         new_seqs.append(tokenizer.convert_ids_to_tokens(seq))
#     return new_seqs

# def make_dict(all_pred,all_target):
#     f1 = 0.
#     for pred,target in zip(all_pred,all_target):
#         pred_srl = []
#         true_srl = []
#         for i,(p_token,t_token) in enumerate(zip(pred,target)):
#             if t_token == "[CLS]" or t_token == "[SEP]":
#                 continue
#             if not t_token == "O":
#                 tag,srl = t_token.split("-")
#                 if not srl.isdecimal():
#                     if tag == "B":
#                         true_dict = {"srl":srl,"start":i}
#     return f1

def get_testseq(x,shuffle,traintest,valtest):
    _,test,_,_ = train_test_split(x,x,shuffle=shuffle,random_state=0,test_size=traintest)
    _,test,_,_ = train_test_split(test,test,shuffle=shuffle,random_state=0,test_size=valtest)
    return test

####################################################
def change_bio2json_opencalm(preds,frame2id,test_json):
    json_list = []
    for idx,pred in enumerate(preds):
        if frame2id:
            json = {"pred_arg":[],"pred_frame":{"start":-1,"end":-1,"frame":-1,"frame_match":False}} # こっちのFrameはB-Pで予測しているからframe_matchは常にFalse
        else:
            json = {"pred_arg":[]}
        role = {}
        sentence = test_json[idx]["sentence"].split() #意味役割の項を取り出しやすくするために文章をListに変換
        for i,token in enumerate(pred):
            if token == "<|endoftext|>":
                continue
            if not (token == "O"): # Bタグまたは Iタグのとき
                bi = token.split("-")[0]
                tag =  token.split("-")[1:]
                tag = "-".join(tag)
                if bi == "B":
                    if tag.isdecimal() or tag == "P": # 述語に関する予測のとき (B-Pのとき)
                        json["pred_frame"]["start"] = i-1 # 開始位置
                        json["pred_frame"]["end"] = i-1 #　終了位置
                        json["pred_frame"]["frame"] = tag #　予測 (P)
                        json["pred_frame"]["surface"] = "".join(sentence[json["pred_frame"]["start"]]) # 該当単語
                        if tag != "P": # 常にPだから 常にFalse
                            #true_frameid = int(test_json[idx]["predicate"]["frameID"].split(":")[0])
                            true_frameid = test_json[idx]["predicate"]["frameID"]
                            if json["pred_frame"]["start"] == test_json[idx]["predicate"]["word_start"] and int(json["pred_frame"]["frame"]) == true_frameid:
                                json["pred_frame"]["frame_match"] = True
                            else:
                                json["pred_frame"]["frame_match"] = False
                    else: # 意味役割に関する予測 (Bタグで意味役割が始まったとき)
                        if role != {}:
                            json["pred_arg"].append(role) # 
                            role = {}
                        role = {"start":i-1,"end":i-1,"role":tag,"surface":"".join(sentence[i-1])} # 意味役割のjson作成 Iタグで続く場合は"end"と"surface"を逐次更新
                else: # Iタグのとき
                    if tag.isdecimal() or tag == "P":
                        json["pred_frame"]["end"] = i-1 # end と surfaceの更新#
                        # json["pred_frame"]["surface"] += " " + sentence[json["pred_frame"]["end"]]
                        json["pred_frame"]["surface"] = "".join(sentence[json["pred_frame"]["end"]]) # 該当単語


                    else:
                        role["end"] = i-1 # endとsurfaceの更新
                        role["surface"] += " " + sentence[i-1]
            else: #終了時点でまだ意味役割がjsonに登録されていない場合は、jsonに追加
                if not role == {}:
                    json["pred_arg"].append(role)
                    role = {}
        json_list.append(json)
        test_json[idx].update(json)
    return test_json

####################################################

def change_bio2json(preds,frame2id,test_json):
    json_list = []
    for idx,pred in enumerate(preds):
        if frame2id:
            json = {"pred_arg":[],"pred_frame":{"start":-1,"end":-1,"frame":-1,"frame_match":False}} # こっちのFrameはB-Pで予測しているからframe_matchは常にFalse
        else:
            json = {"pred_arg":[]}
        role = {}
        sentence = test_json[idx]["sentence"].split() #意味役割の項を取り出しやすくするために文章をListに変換
        for i,token in enumerate(pred):
            if token == "[CLS]":
                continue
            if not (token == "O" or token == "[SEP]"): # Bタグまたは Iタグのとき
                bi = token.split("-")[0]
                tag =  token.split("-")[1:]
                tag = "-".join(tag)
                if bi == "B":
                    if tag.isdecimal() or tag == "P": # 述語に関する予測のとき (B-Pのとき)
                        json["pred_frame"]["start"] = i-1 # 開始位置
                        json["pred_frame"]["end"] = i-1 #　終了位置
                        json["pred_frame"]["frame"] = tag #　予測 (P)
                        json["pred_frame"]["surface"] = "".join(sentence[json["pred_frame"]["start"]]) # 該当単語
                        if tag != "P": # 常にPだから 常にFalse
                            #true_frameid = int(test_json[idx]["predicate"]["frameID"].split(":")[0])
                            true_frameid = test_json[idx]["predicate"]["frameID"]
                            if json["pred_frame"]["start"] == test_json[idx]["predicate"]["word_start"] and int(json["pred_frame"]["frame"]) == true_frameid:
                                json["pred_frame"]["frame_match"] = True
                            else:
                                json["pred_frame"]["frame_match"] = False
                    else: # 意味役割に関する予測 (Bタグで意味役割が始まったとき)
                        if role != {}:
                            json["pred_arg"].append(role) # 
                            role = {}
                        role = {"start":i-1,"end":i-1,"role":tag,"surface":"".join(sentence[i-1])} # 意味役割のjson作成 Iタグで続く場合は"end"と"surface"を逐次更新
                else: # Iタグのとき
                    if tag.isdecimal() or tag == "P":
                        json["pred_frame"]["end"] = i-1 # end と surfaceの更新#
                        json["pred_frame"]["surface"] += " " + sentence[json["pred_frame"]["end"]]

                    else:
                        role["end"] = i-1 # endとsurfaceの更新
                        role["surface"] += " " + sentence[i-1]
            else: #終了時点でまだ意味役割がjsonに登録されていない場合は、jsonに追加
                if not role == {}:
                    json["pred_arg"].append(role)
                    role = {}
        json_list.append(json)
        test_json[idx].update(json)
    return test_json

def calculate_f1(json_list):
    recall = 0
    precision = 0
    all_f1 = 0
    all_precision = 0
    all_recall = 0
    for i,json in enumerate(json_list):
        count = 0
        f1 = 0
        precision = 0
        recall = 0
        trues_len = 0
        for arg in json["args"]: # 正解数のカウント -1 は単語の省略部分だからカウントしない
            if arg["word_start"] != -1:
                trues_len += 1
        json_list[i]["args_len"] = trues_len
        prev_count = 0
        for pred in json["pred_arg"]: # Pは述語の予測だから,そこだけカウント
            if pred["role"] == "P":
                prev_count += 1
        preds_len = len(json["pred_arg"]) - prev_count # モデルの意味役割の予測から述語の予測数だけ減算
        if preds_len == 0: # 予測がゼロのときはF1は0
            f1 = 0
            precision = 0
            recall = 0
            json_list[i]["predicate_len"] = 0
            # json_list[i]["predicate_len"] = 0
            # json_list[i]["precision"] = precision
            # json_list[i]["recall"] = recall
            # json_list[i]["f1"] = f1
        else:
            #print(trues)
            #print(preds)
            predicate_count = 0 # predictと間違えた
            for j,pred in enumerate(json["pred_arg"]):
                start = pred["start"] # 意味役割の開始
                end = pred["end"] # 意味役割の修了
                role = pred["role"] # 意味役割
                if role != "P":
                    predicate_count += 1
                for true in json["args"]:
                    #if true["start"] == start and true["end"] == end and true["arg"] == role:
                    if true["word_start"] == start and true["word_end"] == end and true["argrole"] == role: # 正解と一致したとき
                        count += 1
                        json_list[i]["pred_arg"][j]["true_false"] = True
                        break
                    else: # どれか一つでも間違えたとき
                        json_list[i]["pred_arg"][j]["true_false"] = False
            #print(count)
            if not count == 0:
                json_list[i]["predicate_len"] = predicate_count # predicat_lenと間違えた
                precision = count / preds_len # precision recall F1の計算
                recall = count / trues_len
                f1 = 2*precision*recall / (precision+recall)
                # json_list[i]["precision"] = precision
                # json_list[i]["recall"] = recall
                # json_list[i]["f1"] = f1
            else:
                json_list[i]["predicate_len"] = predicate_count
                precision = 0
                recall = 0
                f1 = 0
                # json_list[i]["precision"] = precision
                # json_list[i]["recall"] = recall
                # json_list[i]["f1"] = f1
        all_precision += precision
        all_recall += recall
        all_f1 += f1
        json_list[i]["match_count"] = count
    print("f1 : ",all_f1/len(json_list)) # データセット全体でのスコアを計算
    print("precision : ",all_precision/len(json_list))
    print("recall : ",all_recall/len(json_list))
    return all_f1/len(json_list),json_list
            


def calculate_json(json_list):
    f1,json_list = calculate_f1(json_list)
    return f1,json_list


def make_bio_in_json(test_json,preds_list):
    pred_bio_tag = []
    # print(test_json[0])
    for i,pred_bio in enumerate(preds_list):
        test_json[i]["BIOtag"] = " ".join(pred_bio[1:-1]) # BIO系列をjsonに追記 clsとsepは除く
        test_json[i]["base_sentence"] = "" # 容量削減のため無駄な系列を消す
        test_json[i]["pos_sentence"] = "" 
    # print(test_json[0])
    return test_json

def chenge_id2predicate(all_preds_list,target_tags):
    new_list = []
    for pred in all_preds_list:
        new_list.append(target_tags[pred])
    return new_list

def change_predicate2json(preds_list,test_json):
    for idx,pred in enumerate(preds_list):
        json_ = {"predict_frame":{}}
        json_["predict_frame"]["frame"] = pred
        true_frameid = test_json[idx]["predicate"]["frameID"]
        if json_["predict_frame"]["frame"] == true_frameid: # 正解と一致しているかの確認用フラグ
            json_["predict_frame"]["frame_match"] = True
        else:
            json_["predict_frame"]["frame_match"] = False
        test_json[idx].update(json_) # データセットのjsonの更新
    return test_json
