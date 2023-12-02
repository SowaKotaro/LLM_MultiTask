# 学習で使うGCN部分は、torch_geometricのGCNConvを使う
# このファイルは、あくまでGCN前後の処理を担う

# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html の手順でcuda, torch のバージョンを確認 (下のtorch-{torchバージョン}, cu{cudaバージョン*10したやつ}, cp{pythonのバージョン})
# ↑に従って torch_geometricまでinstallできたらOK (古いバージョンは↑の中央付近の here のリンクから)
# 以下岩本の手順  (torch_scatterでビルド失敗?した為,AWS経由でinstall)
# (python: 3.6.8, torch: 1.7.1, cuda: 102, aws経由だと1.7.1->1.7.0)
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
# pip install https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.7.0%2Bcu102/torch_scatter-2.0.5-cp36-cp36m-linux_x86_64.whl
# pip install torch-geometric
# 以上で最低限実行可能なはず 以下参考ページ
# https://github.com/pyg-team/pytorch_geometric/issues/2381
# https://stackoverflow.com/questions/60236134/pytorch-geometric-cuda-installation-issues-on-google-colab
# エラーの対処法
# AttributeError: module 'torch.nn.parameter' has no attribute 'UninitializedParameter' のとき 下記のissueで修正されているように
# /home/junki/.pyenv/versions/3.6.8/lib/python3.6/site-packages/torch_geometric/nn/dense/linear.py　を直接修正した
# https://github.com/pyg-team/pytorch_geometric/issues/3439 


import torch
import torch.nn as nn
from torch_scatter import scatter

def get_syntax_info(sentenceid_list,syntax_dic):
    node_batch = []
    edge_batch = []
    tag_batch = []
    
    for sentenceid in sentenceid_list:
        node_batch.append(syntax_dic[sentenceid]["word_position"])
        edge_batch.append(syntax_dic[sentenceid]["edge_rep"])
        tag_batch.extend(syntax_dic[sentenceid]["tag"])
    return node_batch, edge_batch, tag_batch


def token_convert(token_position,sentence_length,cls_bool):  
    reposition = 0
    if cls_bool: # 先頭にCLSがある場合、1単語文位置を修正
        reposition = 1

    # 隠れベクトルのshapeを[batch_size*sentence_length, hidden_size]に変換して使用するときに文章内のノードの先頭単語と末尾単語の位置を取得できるように単語位置のtensorを作成
    word_positions_tensor = torch.tensor(token_position[0]) + reposition
    for i,all_node in enumerate(token_position[1:],1):
        # 前の文章でのsentence_length(padding済)の分のズレを修正しながらノードの先頭単語と末尾単語の位置を作成
        word_positions_tensor = torch.cat((word_positions_tensor,torch.tensor(all_node) + (i * sentence_length) + reposition),dim=0) 
    word_positions_tensor = word_positions_tensor.view(-1) # shapeが[バッチ内で出現する全ノード数,2]を[バッチ内で出現する全ノード数*2]にする
    _, position_index = torch.unique(word_positions_tensor, sorted=True, return_inverse=True) # 
    return word_positions_tensor,position_index

# 文章内のノード全部を一つのベクトルにpoolingする場合の処理 精度低
def token_convert_v2(token_position,sentence_length,cls_bool):  
    reposition = 0
    if cls_bool: # 先頭にCLSがある場合、その分だけ位置を修正
        reposition = 1
    word_positions_tensor = torch.tensor(token_position[0]) + reposition
    sentence_positions = torch.full((len(token_position[0]),2),fill_value=0)
    for i,all_node in enumerate(token_position[1:],1):
        word_positions_tensor = torch.cat((word_positions_tensor,torch.tensor(all_node) + (i * sentence_length) + reposition),dim=0)
        sentence_positions = torch.cat((sentence_positions, torch.full((len(all_node),2),fill_value=i)),dim=0)
    word_positions_tensor = word_positions_tensor.view(-1)
    sentence_positions = sentence_positions.view(-1)
    return word_positions_tensor, sentence_positions


def gcn_edge_tensor(edge_batch):
    edge_tensor = torch.tensor(edge_batch[0],dtype=torch.long) 
    if len(edge_batch[0][0]):
        next_start_index = edge_batch[0][0][-1] + 1 # エッジの送信側が昇順に設定されている前提
    else: # 短い文章でroot node(IP-MAT)しかない場合はエッジがないため
        next_start_index = 1
    for i,edge_list in enumerate(edge_batch[1:],1): # バッチサイズ分の出現ノードに対して、前のノード数を足してノードのインデックスの位置を修正しながら連結
        edge_tensor = torch.cat((edge_tensor,torch.tensor(edge_list,dtype=torch.long) + next_start_index),dim=1)
        if len(edge_list[0]):
            next_start_index += edge_list[0][-1] + 1
        else:
            next_start_index += 1
    return edge_tensor

# one hotで連結する場合
def make_tag_onehot(node_tags,tag_length):
    tag_onehot = torch.zeros(len(node_tags),tag_length,dtype=torch.uint8)
    for i,tag_idx in enumerate(node_tags):
        tag_onehot[i][tag_idx] = 1 
    return tag_onehot

def make_tag_num(node_tags):
    tag_tensor = torch.zeros(len(node_tags),dtype=torch.long) # zerosだけど 結局全部 tag idに置き換わる
    for i,tag_id in enumerate(node_tags): # それぞれのタグについてtag_id変換
        tag_tensor[i] = tag_id 
    return tag_tensor


class SeqToNode(nn.Module):
    def __init__(self,mode="cat"):
        super().__init__()
        self.mode = mode

    def forward(self,hidden,word_absolute_position):
        # hidden shape [batch_size, sentence_length, hidden_size] を [batch_size * sentence_length, hidden_size] に変換
        reshape_hidden = hidden.view(hidden.shape[0]*hidden.shape[1],hidden.shape[2])
        token_hidden = torch.index_select(reshape_hidden,0,word_absolute_position) # reshape_hiddenからノードに使用する単語を一括取得
        if self.mode == "cat":
            token_hidden = token_hidden.view(-1,2*hidden.shape[2]) # 連結してる状態になる
        elif self.mode == "add":
            token_hidden = token_hidden[::2] + token_hidden[1::2]
        return token_hidden


# 処理イメージのリンク(https://docs.google.com/presentation/d/1gahDd4_PpPKTxyoLkaNjYMH6VNDNV6y32GDmgZpkOwo/edit#slide=id.g20f2319fc3f_0_17)
class NodeToSeq(nn.Module):
    def __init__(self,device,mode="sum"):
        super().__init__()
        self.device = device
        self.mode = mode

    def forward(self,gcn_hidden,word_absolute_position,word_index_position,batch_size,max_length):
        gcn_hidden = gcn_hidden.unsqueeze(1)
        gcn_hidden = gcn_hidden.repeat(1,2,1)  # ノードの特徴量を2つに複製 (scatterで一括抽出するため)
        gcn_hidden = gcn_hidden.view(gcn_hidden.shape[0]*2,gcn_hidden.shape[2])  # [ノード数,2,隠れベクトル数]を [ノード数*2,隠れベクトル数]
        pooled_node_hidden = scatter(gcn_hidden,word_index_position,dim=0,reduce=self.mode)  # ノードの開始単語、末尾単語に対して、適用したノード全部をreduceの設定によってsum(or mean or ...) pooling
        zeros = torch.zeros(batch_size*max_length,gcn_hidden.shape[1]).to(self.device) # ノード毎のベクトルから単語毎のベクトルに連結するためzero tensorで初期化
        zeros[word_absolute_position,] = torch.index_select(pooled_node_hidden,0,word_index_position) # poolingしたノードベクトルを単語の位置に展開
        zeros = zeros.view(batch_size,max_length,gcn_hidden.shape[1]) # [batch_size, sentence_length, node_hidden]に変換 これをBERTの出力に連結
        return zeros

# 文章内のノード全部を一つのベクトルにpoolingする場合の処理 精度低
class NodeToSeq_v2(nn.Module):
    def __init__(self,device,mode="sum"):
        super().__init__()
        self.device = device
        self.mode = mode

    def forward(self,gcn_hidden,token_position,batch_size,max_length,token_position2):
        gcn_hidden = gcn_hidden.unsqueeze(1)
        gcn_hidden = gcn_hidden.repeat(1,2,1)  # 複製
        gcn_hidden = gcn_hidden.view(gcn_hidden.shape[0]*2,gcn_hidden.shape[2])  # 参照
        x = scatter(gcn_hidden,token_position,dim=0,reduce=self.mode)
        x = x.unsqueeze(1).expand(batch_size,max_length,gcn_hidden.shape[1])
        # print(x.shape)
        return x