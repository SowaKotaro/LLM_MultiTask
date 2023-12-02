import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from preprocessing import prepro
from preprocessing.pp_bert import json_encode_bert
from preprocessing.pp_opencalm import json_encode_opencalm

from network import multi_task_BP_simple
import json

# MODEL = "BERT"
# MODEL = "OPENCALM-SMALL"
MODEL = "OPENCALM-MEDIUM"

def main():
    params = prepro.read_params("parameter_multi_task")  # パラメータの読み込み バッチサイズとか epoch数とか
    # with open("./../data/dataset/data_v2.json") as f: # データセットの読み込み
    #     data = json.load(f)
    # print(params)
    with open("./../data/dataset/NonHash_data_v2.json") as f: # データセットの読み込み
        data = json.load(f)
    # conll2json.pyによって得られるsyntax.jsonに対して、同ディレクトリにあるget_flat_json.pyを実行することで得られるflatな構文構造を使用
    with open("./../data/dataset/flat_syntax_v2.json",mode="r") as f: # 構文構造の読み込み
        syntax_json = json.load(f)

    # data = prepro.sentence_shuffle(data)
    if MODEL == "BERT":
        inputs, bio_targets, fid_targets, arg_tags, _, fid_candidate = json_encode_bert.encode_multi(data, params) # jsonデータセットから学習に必要な 文章 BIOの出力, argsの数, FIDの候補 を返す
        syntax_dic,syntaxid2tag = json_encode_bert.syntax_encode(syntax_json,params["use_parents_node"],params["use_child_node"]) # 構文構造のjsonから学習に利用しやすい形式で必要な情報を取り出す
    elif "OPENCALM" in MODEL:
        inputs, bio_targets, fid_targets, arg_tags, _, fid_candidate = json_encode_opencalm.encode_multi_opencalm(data, params) # jsonデータセットから学習に必要な 文章 BIOの出力, argsの数, FIDの候補 を返す
        syntax_dic,syntaxid2tag = json_encode_opencalm.syntax_encode_opencalm(syntax_json,params["use_parents_node"],params["use_child_node"]) # 構文構造のjsonから学習に利用しやすい形式で必要な情報を取り出す

    # inputs, bio_targets, fid_targets, arg_tags, _, fid_candidate = json_encode.encode_multi(data, params) # jsonデータセットから学習に必要な 文章 BIOの出力, argsの数, FIDの候補 を返す
    # syntax_dic,syntaxid2tag = json_encode.syntax_encode(syntax_json,params["use_parents_node"],params["use_child_node"]) # 構文構造のjsonから学習に利用しやすい形式で必要な情報を取り出す
    
    # SRLシンプルモデル(BPと表記)との組み合わせ
    # BP simple
    multi_task_BP_simple.larning_process(inputs, bio_targets, fid_targets, fid_candidate, data, arg_tags, syntax_dic, syntaxid2tag, params, MODEL)
    # BP input candidate
    # multi_task_BP_input.larning_process(inputs, bio_targets, fid_targets, fid_candidate, data, arg_tags, syntax_dic, syntaxid2tag, params)
    # BP Limited
    # multi_task_BP_limited.larning_process(inputs, bio_targets, fid_targets, fid_candidate, data, arg_tags, syntax_dic, syntaxid2tag, params)

if __name__ == "__main__":
    main()
