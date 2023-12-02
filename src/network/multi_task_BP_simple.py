import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from transformers import BertModel
from torch_geometric.nn import GCNConv
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from preprocessing import prepro
from preprocessing.pp_bert import encode_bert
from preprocessing.pp_opencalm import encode_opencalm
from . import earlystop, padding, decoder, crf, gcn_module
import pickle
import json
from transformers import GPTNeoXModel, GPTNeoXPreTrainedModel, AutoTokenizer

FILE_NAME = "multitask_BP2_Simple_v2data"
MECAB_DICT_PATH = "./../data/unidic"


######################################################
class MultiTask_opencalm(nn.Module):
    def __init__(self,bio_length,fid_length,syntax_tag_length,transition,device,MODEL):
        super().__init__()
        if "SMALL" in MODEL:
            self.opencalm = GPTNeoXModel.from_pretrained("cyberagent/open-calm-small")
        elif "MEDIUM" in MODEL:
            self.opencalm = GPTNeoXModel.from_pretrained("cyberagent/open-calm-medium")
        
        dim = self.opencalm.config.hidden_size
        
        self.average_pooling = AveragePooling()
        self.dence = nn.Linear(dim*2,fid_length)
        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden2tag = nn.Linear(dim+1,500)
        self.hidden2tag2 = nn.Linear(500,bio_length)
        self.relu = nn.ReLU()
        self.crf = crf.CRF(bio_length,batch_first=True)
        self.crf.set_transitions(transition)

    def get_eos_index_opencalm(self,x):
        tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small") # スペシャルトークンを取り出す．別段smallである意味はない
        eos_token = tokenizer.eos_token_id
        # print("eos_token",eos_token)
        eos_index_list = []
        for i in range(len(x)):
            row = x[i,:]
            # print("row",row)
            eos_index = [i for i, id in enumerate(row) if id == eos_token]
            # print("eos_index type",type(eos_index))
            eos_index_list.append(eos_index[1]) # 2番目の0の位置を格納
            # print("eos_index_LIST",eos_index_list)
        return eos_index_list


    def get_emission(self,x,position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,predicate_position):        
        h = self.opencalm(x,x_mask)
        eos_index_list = []       
        batch_size = len(x[:])
        dim = self.opencalm.config.hidden_size

        average = self.average_pooling(h["last_hidden_state"],predicate_position)
        eos_index_list = self.get_eos_index_opencalm(x)
        eos_embedding = torch.zeros(batch_size,dim).cuda() # .cuda()無しだとCPU上にテンソルを作成し，torch.cat出来なくなる

        for i in range(batch_size):
            eos_embedding[i,:] = h["last_hidden_state"][i,eos_index_list[i],:]

        fid_cat_h = torch.cat((eos_embedding,average),dim=1) # [32, dim]⊕[32, dim]=>[32, dim*2]
        fid_output = self.dence(fid_cat_h)
        fid_output = self.softmax(fid_output)
        
        srl_cat_h = torch.cat((h["last_hidden_state"],position),dim=2)
        srl_output = self.hidden2tag(srl_cat_h)
        srl_output = self.relu(srl_output)
        srl_output = self.hidden2tag2(srl_output)
        srl_output = self.relu(srl_output)
        # exit(0)
        return fid_output, srl_output
    

    def loss_pred(self,x,y,position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_token_type,predicate_position):
        fid_output, srl_output = self.get_emission(x,position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag, predicate_position)
        
        loss = -self.crf(srl_output,y,x_mask^x_token_type)
        pred = self.crf.decode(srl_output,x_mask^x_token_type)

        return loss, pred, fid_output
    
    def decode(self,x,position,x_mask):
        emissions = self.get_emission(x,position,x_mask)
        best_tags = self.crf.decode(emissions,x_mask)
        return best_tags
######################################################
class MultiTask_BP_Simple(nn.Module):
    def __init__(self,bio_length,fid_length,syntax_tag_length,transition,device):
        super().__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')

        # self.embedding = nn.Embedding(syntax_tag_length,60) # 構文タグのidを60次元の分散表現に拡張
        # self.gcn1 = GCNConv(60,100)
        # self.node2seq = gcn_module.NodeToSeq(device, "sum")
        # self.lstm = nn.LSTM(100,100,batch_first=True,bidirectional=True)

        self.averege_pooling = AveragePooling()
        # print(fid_length)
        self.dence = nn.Linear(768*2,fid_length)
        self.softmax = nn.LogSoftmax(dim=1)

        # print("biolength:",bio_length)
        self.hidden2tag = nn.Linear(768+1,500) # 500
        self.hidden2tag2 = nn.Linear(500,bio_length)
        self.relu = nn.ReLU()
        self.crf = crf.CRF(bio_length,batch_first=True)
        self.crf.set_transitions(transition)
    
    def get_emission(self,x,position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,predicate_position):

        h = self.bert(x,x_mask)

        # node_embedding = self.embedding(syntax_tag)
        # gcn_hidden = self.gcn1(node_embedding,edge_index=edge_tensor)
        # gcn_word_hidden = self.node2seq(gcn_hidden,word_absolute_position,word_index_position,h["last_hidden_state"].shape[0],h["last_hidden_state"].shape[1])
        # gcn_word_hidden, _ = self.lstm(gcn_word_hidden)

        # cat_h = torch.cat((h["last_hidden_state"],gcn_word_hidden),dim=2) 
        average = self.averege_pooling(h["last_hidden_state"],predicate_position)
        fid_cat_h = torch.cat((h["last_hidden_state"][:,0,:],average),1)
        fid_output = self.dence(fid_cat_h)
        fid_output = self.softmax(fid_output)

        srl_cat_h = torch.cat((h["last_hidden_state"],position),dim=2)
        srl_output = self.hidden2tag(srl_cat_h)
        srl_output = self.relu(srl_output)
        srl_output = self.hidden2tag2(srl_output)
        srl_output = self.relu(srl_output)

        return fid_output, srl_output
    

    def loss_pred(self,x,y,position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_token_type,predicate_position):
        fid_output, srl_output = self.get_emission(x,position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag, predicate_position)
        
        loss = -self.crf(srl_output,y,x_mask^x_token_type)
        pred = self.crf.decode(srl_output,x_mask^x_token_type)

        # print("$$$",loss)
        # print("!!!",pred)
        # print("|||",fid_output)
        return loss, pred, fid_output
    
    def decode(self,x,position,x_mask):
        emissions = self.get_emission(x,position,x_mask)
        best_tags = self.crf.decode(emissions,x_mask)
        return best_tags

class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,h, target_position): 
        product = torch.mul(h,target_position) # アダマール積
        product = product.sum(1) / target_position.sum(1)
        return product

def larning_process(wakati, bio_targets, fid_targets, fid_candidate, jsondata, arg_tags, syntax_dic, syntaxid2tag, params, MODEL):
    torch.manual_seed(111)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # エンコード ID化
    if MODEL == "BERT":
        sentence_id_list = encode_bert.get_sentenceid(jsondata)
        wakati_id = encode_bert.seq2bertid(wakati, MECAB_DICT_PATH)
        fid_tags = prepro.frameid2tag()
        fid_targets_id,fid_candidate_id = encode_bert.target2id_v2(fid_targets, fid_tags, fid_candidate)
        bio_targets_id,id2bio,transition = encode_bert.targetseq2id(bio_targets,arg_tags,params["not_O_fidtag"],params["predicate_is_BP"])
    elif "OPENCALM" in MODEL:
        sentence_id_list = encode_opencalm.get_sentenceid_opencalm(jsondata)
        wakati_id = encode_opencalm.seq2id_opencalm(wakati, MECAB_DICT_PATH, MODEL)
        fid_tags = prepro.frameid2tag()
        fid_targets_id,fid_candidate_id = encode_opencalm.target2id_v2_opencalm(fid_targets, fid_tags, fid_candidate)
        bio_targets_id,id2bio,transition = encode_opencalm.targetseq2id_opencalm(bio_targets,arg_tags,params["not_O_fidtag"],params["predicate_is_BP"])


    # ndarrayA = np.array(wakati_id)
    # ndarrayB = np.array(fid_targets_id)
    # ndarrayC = np.array(bio_targets_id)
    # ndarrayD = np.array(fid_candidate_id)
    # ndarrayE = np.array(sentence_id_list)
    # ndarrayV = np.array(x_test)
    # ndarrayW = np.array(t_fid_test)
    # ndarrayX = np.array(t_srl_test)
    # ndarrayY = np.array(fid_candidate_test)
    # ndarrayZ = np.array(sentenceid_test)
    # print("wakati_id shape",ndarrayA.shape)
    # print("fid_target_id shape",ndarrayB.shape)
    # print("bio_targets_id shape",ndarrayC.shape)
    # print("fid_candidate_id shape",ndarrayD.shape)
    # print("sentence_id_list shape",ndarrayE.shape)
    # print("="*20)
    # print("x_test shape",ndarrayV.shape)
    # print("t_fid_test shape",ndarrayW.shape)
    # print("t_srl_test shape",ndarrayX.shape)
    # print("fid_candidate_test shape",ndarrayY.shape)
    # print("sentence_id_test shape",ndarrayZ.shape)


    x_train, x_test, t_fid_train, t_fid_test, t_srl_train, t_srl_test, fid_candidate_train, fid_candidate_test, sentenceid_train, sentenceid_test = train_test_split(wakati_id,fid_targets_id,bio_targets_id,fid_candidate_id,sentence_id_list,shuffle=params["shuffle"],test_size=params["train_tv_size"]) # shuffle -> False, train_tv_size -> 0.2
    x_val, x_test, t_fid_val, t_fid_test, t_srl_val, t_srl_test, fid_candidate_val, fid_candidate_test, sentenceid_val, sentenceid_test = train_test_split(x_test, t_fid_test, t_srl_test, fid_candidate_test, sentenceid_test, shuffle=params["shuffle"], test_size=params["valid_test_size"]) #  shuffle -> False, valid_test_size -> 0.5
    # print("dou?")

    # reporter = MemReporter(bert_fid)
    if MODEL == "BERT":
        print("select model:",MODEL)
        multi_task = MultiTask_BP_Simple(len(id2bio),len(fid_tags), len(syntaxid2tag),transition,device).to(device)
    elif "OPENCALM" in MODEL:
        print("select model:",MODEL)
        multi_task = MultiTask_opencalm(len(id2bio),len(fid_tags), len(syntaxid2tag),transition,device,MODEL).to(device)

    if "OPENCALM" in MODEL:
        final_layer_index = multi_task.opencalm.config.num_hidden_layers - 1
        # 全パラメータ更新
        # for name, param in multi_task.opencalm.named_parameters():
        #     param.requires_grad = True

        # 最終層のパラメータのみ更新
        for name, param in multi_task.opencalm.named_parameters():
            param.requires_grad = False # 一旦全てFalse 
        for name, param in multi_task.opencalm.layers[final_layer_index].named_parameters():
            param.requires_grad = True  # 最終層のみTrue

    else:
        # 最終層以外のパラメータ固定
        for name, param in multi_task.bert.named_parameters():
            param.requires_grad = False
        # for name, param in bert_fid.bert.encoder.layer[9].named_parameters():
        #     param.requires_grad = True
        # for name, param in bert_fid.bert.encoder.layer[10].named_parameters():
        #     param.requires_grad = True
        for name, param in multi_task.bert.encoder.layer[11].named_parameters():
            param.requires_grad = True
        for name, param in multi_task.bert.pooler.named_parameters():
            param.requires_grad = True
    
    optimizer = opt.Adam(multi_task.parameters())
    #cal_loss = nn.CrossEntropyLoss()
    cal_loss = nn.NLLLoss()
    earlystopping = earlystop.EarlyStopping(path='./checkpoint/{}.pth'.format(FILE_NAME), patience=5)

    def fid_calculate_accuracy(t, preds):
        # preds = preds.to('cpu').detach().numpy()
        accuracy_c = 0.
        for target, a_batch in zip(t, preds):
            argmax = torch.argmax(a_batch)
            if argmax == target:
                accuracy_c += 1
        accuracy = accuracy_c / preds.shape[0]
        return accuracy
    
    def srl_calculate_accuracy(t,pred):
        acc = 0.
        for i in range(len(t)):
            acc += accuracy_score(t[i][0], pred[i])
        acc = acc / len(t)
        return acc

    def train_step(x_seq,srl_t,frame_position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_token_type,predicate_position,fid_t):
        x_seq = torch.LongTensor(x_seq).to(device)
        x_mask = torch.ByteTensor(x_mask).to(device)
        x_token_type = torch.LongTensor(x_token_type).to(device)
        predicate_position = torch.ByteTensor(predicate_position).to(device)
        frame_position = torch.LongTensor(frame_position).to(device)

        word_absolute_position = torch.LongTensor(word_absolute_position).to(device)
        word_index_position = torch.LongTensor(word_index_position).to(device)
        edge_tensor = torch.LongTensor(edge_tensor).to(device)
        syntax_tag = torch.LongTensor(syntax_tag).to(device)
        gpu_t = torch.LongTensor(srl_t).to(device)    
        fid_t = torch.LongTensor(fid_t).to(device)

        multi_task.train()
        srl_loss, srl_pred, fid_pred = multi_task.loss_pred(x_seq,gpu_t,frame_position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_token_type,predicate_position)
        
        fid_loss = cal_loss(fid_pred,fid_t)
        fid_loss /= x_seq.shape[0]
        srl_loss /= x_seq.shape[0]
        loss = fid_loss*10 + srl_loss

        multi_task.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, srl_pred, fid_pred

    def val_step(x_seq,srl_t,frame_position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_token_type,predicate_position,fid_targets):
        x_seq = torch.LongTensor(x_seq).to(device)
        x_mask = torch.ByteTensor(x_mask).to(device)
        x_token_type = torch.LongTensor(x_token_type).to(device)
        predicate_position = torch.ByteTensor(predicate_position).to(device)
        frame_position = torch.LongTensor(frame_position).to(device)

        word_absolute_position = torch.LongTensor(word_absolute_position).to(device)
        word_index_position = torch.LongTensor(word_index_position).to(device)
        edge_tensor = torch.LongTensor(edge_tensor).to(device)
        syntax_tag = torch.LongTensor(syntax_tag).to(device)
    
        gpu_t = torch.LongTensor(srl_t).to(device)    
        fid_targets = torch.LongTensor(fid_targets).to(device)

        multi_task.eval()
        srl_loss, srl_pred, fid_pred = multi_task.loss_pred(x_seq,gpu_t,frame_position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_token_type,predicate_position)

        fid_loss = cal_loss(fid_pred,fid_targets)
        fid_loss /= x_seq.shape[0]
        srl_loss /= x_seq.shape[0]
        loss = fid_loss*10 + srl_loss

        return loss, srl_pred, fid_pred

    # 一旦、トークンタイプは全部ゼロ
    epochs = params['epochs']
    batch_size = params['batch_size']
    n_batches_train = len(x_train)//batch_size
    n_batches_val = len(x_val)//batch_size
    print("train batch", n_batches_train)
    print("valid batch", n_batches_val)

    print("train start epochs:{}".format(epochs))
    # bert_crf.load_state_dict(torch.load('./checkpoint/bertcrf_v4_BP_Bframeid_es3.pth'))  # BP best score: bertcrf_v4_BP.pth
    for epoch in range(epochs):
        # if epoch < 11:  # 途中からスタートする時
        #     continue
        train_loss = 0.
        train_fid_acc = 0.
        train_srl_acc = 0.
        val_loss = 0.
        val_srl_acc = 0.
        val_fid_acc = 0.
        # x_, t_ = shuffle(x_train, t_train)
        x_, t_fid_, t_srl_, fid_candidate_, sentenceid_  = shuffle(x_train, t_fid_train, t_srl_train, fid_candidate_train, sentenceid_train)
        #x_,t_ = x_train,t_train
        for batch in range(n_batches_train):
            start = batch*batch_size
            end = start+batch_size

            x_seq = padding.seq_padding_v2(x_[start:end])
            x_mask = padding.make_mask(x_[start:end], params['batch_size'], x_seq.shape[1])
            if "OPENCALM" in MODEL:
                x_token_type = padding.make_token_type_opencalm(x_[start:end],params['batch_size'], x_seq.shape[1])
                predicate_position = padding.make_predicate_position_opencalm(x_[start:end], params['batch_size'], x_seq.shape[1])
            elif MODEL == "BERT":
                x_token_type = padding.make_token_type(x_[start:end],params['batch_size'], x_seq.shape[1])
                predicate_position = padding.make_predicate_position(x_[start:end], params['batch_size'], x_seq.shape[1])

            frame_position = padding.make_frameposition(t_srl_[start:end],x_seq.shape[1])

            # バッチ毎の構文構造取得に関する処理
            node_batch,edge_batch,tag_batch = gcn_module.get_syntax_info(sentenceid_[start:end],syntax_dic)
            word_absolute_position,word_index_position = gcn_module.token_convert(node_batch,x_seq.shape[1],params["cls"])
            edge_tensor = gcn_module.gcn_edge_tensor(edge_batch)
            syntax_tag = gcn_module.make_tag_num(tag_batch)

            fid_t = torch.tensor(t_fid_[start:end], dtype=torch.long)
            srl_t = padding.seq_padding_v3(t_srl_[start:end],x_seq.shape[1])
            loss, srl_pred, fid_pred = train_step(x_seq,srl_t,frame_position,x_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_token_type,predicate_position,fid_t)
            train_loss += loss.item()

            train_fid_acc += fid_calculate_accuracy(fid_t,fid_pred)
            train_srl_acc += srl_calculate_accuracy(t_srl_[start:end],srl_pred)
        train_loss /= n_batches_train
        train_fid_acc /= n_batches_train
        train_srl_acc /= n_batches_train

        print("validate")
        for batch in range(n_batches_val):
            start = batch * batch_size
            end = start + batch_size

            x_val_seq = padding.seq_padding_v2(x_val[start:end])
            x_val_mask = padding.make_mask(x_val[start:end], params['batch_size'], x_val_seq.shape[1])
            if "OPENCALM" in MODEL:
                x_val_token_type = padding.make_token_type_opencalm(x_val[start:end],params['batch_size'], x_val_seq.shape[1])
                val_predicate_position = padding.make_predicate_position_opencalm(x_val[start:end], params['batch_size'], x_val_seq.shape[1])
            elif MODEL == "BERT":
                x_val_token_type = padding.make_token_type(x_val[start:end],params['batch_size'], x_val_seq.shape[1])
                val_predicate_position = padding.make_predicate_position(x_val[start:end], params['batch_size'], x_val_seq.shape[1])
        
            val_frame_position = padding.make_frameposition(t_srl_val[start:end],x_val_seq.shape[1])

            # バッチ毎の構文構造取得に関する処理
            node_batch,edge_batch,tag_batch = gcn_module.get_syntax_info(sentenceid_val[start:end],syntax_dic)
            word_absolute_position,word_index_position = gcn_module.token_convert(node_batch,x_val_seq.shape[1],params["cls"])
            edge_tensor = gcn_module.gcn_edge_tensor(edge_batch)
            syntax_tag = gcn_module.make_tag_num(tag_batch)
            
            val_fid_t = torch.tensor(t_fid_val[start:end], dtype=torch.long)
            val_srl_t = padding.seq_padding_v3(t_srl_val[start:end],x_val_seq.shape[1])

            loss, srl_pred, fid_pred = val_step(x_val_seq,val_srl_t,val_frame_position,x_val_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_val_token_type,val_predicate_position,val_fid_t)
            val_loss += loss.item()

            val_fid_acc += fid_calculate_accuracy(val_fid_t,fid_pred)
            val_srl_acc += srl_calculate_accuracy(t_srl_val[start:end], srl_pred)
        val_loss /= n_batches_val
        val_fid_acc /= n_batches_val
        val_srl_acc /= n_batches_val

        string = 'epoch: {},loss: {:.5}, fid_acc:{:.5f}, srl_acc{:.5f}, val_loss: {:.5}, fid_acc:{:.5f}, srl_acc{:.5f}'.format(
            epoch+1, train_loss, train_fid_acc, train_srl_acc, val_loss, val_fid_acc, val_srl_acc)
        print(string)

        earlystopping(val_loss, multi_task)  # callメソッド呼び出し
        if earlystopping.early_stop:  # ストップフラグがTrueの場合、breakでforループを抜ける
            print("Early Stopping!")
            break

    print("test start")
    multi_task.load_state_dict(torch.load('./checkpoint/{}.pth'.format(FILE_NAME)))

    test_loss = 0.
    test_fid_acc = 0.
    test_srl_acc = 0.
    all_preds_fid = []
    all_preds_srl = []
    
    for idx in range(len(x_test)):
        x_test_seq = padding.seq_padding_v2([x_test[idx]])
        x_test_mask = padding.make_mask(x_test[idx], params['batch_size'], x_test_seq.shape[1],test=True)
        if "OPENCALM" in MODEL:
            x_test_token_type = padding.make_token_type_opencalm([x_test[idx]],1, x_test_seq.shape[1])
            test_predicate_position = padding.make_predicate_position_opencalm([x_test[idx]], 1, x_test_seq.shape[1])
        elif MODEL == "BERT":
            x_test_token_type = padding.make_token_type([x_test[idx]],1, x_test_seq.shape[1])
            test_predicate_position = padding.make_predicate_position([x_test[idx]], 1, x_test_seq.shape[1])
        
        test_frame_position = padding.make_frameposition([t_srl_test[idx]], x_test_seq.shape[1])
        
        # バッチ毎の構文構造取得に関する処理
        node_batch,edge_batch,tag_batch = gcn_module.get_syntax_info([sentenceid_test[idx]],syntax_dic)
        word_absolute_position,word_index_position = gcn_module.token_convert(node_batch,x_test_seq.shape[1],params["cls"])
        edge_tensor = gcn_module.gcn_edge_tensor(edge_batch)
        syntax_tag = gcn_module.make_tag_num(tag_batch)
        
        test_fid_t = torch.tensor([t_fid_test[idx]], dtype=torch.long)
        test_srl_t = padding.seq_padding_v3([t_srl_test[idx]],x_test_seq.shape[1])
        
        loss, srl_pred, fid_pred = val_step(x_test_seq,test_srl_t,test_frame_position,x_test_mask,word_absolute_position,word_index_position,edge_tensor,syntax_tag,x_test_token_type,test_predicate_position,test_fid_t)
        test_loss += loss.item()
        
        test_fid_acc += fid_calculate_accuracy(test_fid_t, fid_pred)
        test_srl_acc += srl_calculate_accuracy([t_srl_test[idx]], srl_pred)
        all_preds_fid.append(torch.argmax(fid_pred[0]))
        all_preds_srl.append(srl_pred[0])

    test_loss /= len(x_test)
    test_srl_acc /= len(x_test)
    test_fid_acc /= len(x_test)
    string = 'test_loss: {:.5},test_fid_acc:{:.5f}, test_srl_acc:{:.5f}'.format(test_loss,test_fid_acc,test_srl_acc)
    print(string)

    
    test_json = decoder.get_testseq(jsondata,params["shuffle"],params["train_tv_size"],params["valid_test_size"])
    
    all_preds_fid = decoder.chenge_id2predicate(all_preds_fid,fid_tags)
    test_json = decoder.change_predicate2json(all_preds_fid,test_json)
    
    all_preds_srl = decoder.chenge_id2bio(all_preds_srl,id2bio)
    test_json = decoder.make_bio_in_json(test_json,all_preds_srl)
    if "OPENCALM" in MODEL:
        test_json = decoder.change_bio2json_opencalm(all_preds_srl,params["not_O_fidtag"],test_json)
    elif MODEL == "BERT":
        test_json = decoder.change_bio2json(all_preds_srl,params["not_O_fidtag"],test_json)
    _, test_json = decoder.calculate_json(test_json)
    

    
    with open("./../data/output_multitask/{}.json".format(FILE_NAME),mode="w") as f:
        json.dump(test_json,f,indent=2,ensure_ascii=False)
