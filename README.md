# LLM_MultiTask

LLM_MultiTask
|- README.md
|
|- src
|   |- main_multi_task.py
|   |
|   |- checkpoint
|   |   |-multitask_BP2_Simple_v2data.pth
|   | 
|   |- network
|   |   |- crf.py
|   |   |- earlystop.py
|   |   |- multi_task_BP_simple.py
|   |   |- decoder.py
|   |   |- gcn_module.py
|   |   |- padding.py
|   |   |
|   |   |- __pycache__
|   |
|   |- preprocessing
|       |- prepro.py
|       |
|       |- pp_bert
|       |   |- encode_bert.py
|       |   |- json_encode_bert.py
|       |   |
|       |   |- __pycache__
|       |
|       |- pp_opencalm
|       |   |- encode_opencalm.py
|       |   |- json_encode_opencalm.py
|       |   |
|       |   |- __pycache__
|       |
|       |- __pycache__
|
|- data