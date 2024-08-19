# LearnRepr

## Environment
```
source activate; conda activate /vepfs/zekai/opt/conda/envs/repr310; cd /vepfs/wcf/G/zekai/root/.code/LearnRepr;
```
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Logs
https://docs.qq.com/doc/DVktla2xUYXRzV0pO

## TODO
# gpt4o
- ```python exp/run_proxy_compress.py --task=tensorflow_icl1 --embedding=gte-small --num_epoch=5 --do_eval=0```
- ```python exp/run_proxy_compress.py --task=tensorflow_icl2 --embedding=gte-small --num_epoch=5 --do_eval=1```
- ```python exp/run_proxy_compress.py --task=tensorflow --embedding=gte-small --num_epoch=10 --do_eval=1```
- ```python exp/run_proxy_llm.py --task=tensorflow --embedding=gte-small --num_epoch=10 --do_eval=1```
- ```python exp/run_proxy.py --task=tensorflow --embedding=gte-small --num_epoch=10 --do_eval=1```
- ```python exp/run_proxy.py --task=tensorflow --embedding=gte-small --num_epoch=10 --do_eval=1 --do_train=0```
- ```python exp/run_proxy_llm_compress_last_epoch.py --task=tensorflow --embedding=gte-small --num_epoch=10 --do_eval=0```


# Now all using gpt4o!!

- ```python exp/run_proxy_compress.py --task=tensorflow_icl2 --embedding=gte-small --num_epoch=5 --do_eval=1 --do_train=0 > logs/proxy_compress_icl_gpt4o.txt```
- ```python exp/run_proxy.py --task=tensorflow --embedding=gte-small --num_epoch=10 --do_eval=1 --do_train=0 > logs/proxy_gpt4o.txt```
- ```python exp/run_proxy_llm.py --task=tensorflow --embedding=gte-small --num_epoch=10 --do_eval=1 --do_train=0 > logs/proxy_gpt4o.txt```

环境配置
source activate; conda activate /vepfs/zekai/opt/conda/envs/repr310; cd /vepfs/wcf/G/zekai/root/.code/LearnRepr/
python trial.py


# Use GPT4o! Save to output_v2, Log to logs, add eval
7.23
[1] ape with gte 10 epoch
python exp/run_ape.py > logs_v2/ape.txt
[2] proxy v1 10 epoch
python exp/run_proxy.py > logs_v2/proxy_v1.txt
[3] proxy v1 + compress 10 epoch
python exp/run_proxy_compress.py > logs_v2/proxy_v1_compress.txt


Feedback:似乎所有的只有前5个epoch有用，所以需要inference一下epoch=5
7.24

# Inference
[1] python exp/run_ape.py --num_epoch=5 --do_train=0 > logs_v2/ape_epoch5.txt
[2] python exp/run_proxy.py --num_epoch=5 --do_train=0 > logs_v2/proxy_v1_epoch5.txt
[3] python exp/run_proxy_compress.py --num_epoch=5 --do_train=0 > logs_v2/proxy_v1_compress_epoch5.txt

# proxy v1 + compress icl1 for 10% apis 5 epochs
[6] python exp/run_proxy_compress_icl1.py --num_epoch=2 > logs_v2/proxy_v1_compress_icl1.txt 
python src/zeroshot_initialize.py
python exp/run_proxy_compress_icl2.py --num_epoch=3 > logs_v2/proxy_v1_compress_icl2.txt 


# proxy v2 (load from proxy v1 4 epoch + 1 epoch)
[4] python exp/run_proxy_llm.py > logs_v2/proxy_v2.txt
[4'] python exp/run_proxy_llm.py > logs_v2/proxy_v2_2.txt
# proxy v2 + compress (load from proxy v1 compress 4 epoch + 1 epoch)
[5] python exp/run_proxy_llm_compress_all_epoch.py > logs_v2/proxy_v2_compress.txt
[5'] python exp/run_proxy_llm_compress_all_epoch.py > logs_v2/proxy_v2_compress_2.txt

# proxy v2 + compress + icl (4 epoch + 1 epoch)
[7] python exp/run_proxy_llm_compress_icl_all_epoch.py > logs_v2/proxy_v2_compress_icl.txt

7.25
[1] Eval proxy v2 (because it uses proxy llm and use llama3)
python exp/run_proxy_llm.py --do_train=0 --choice_model=gpt4o > logs_v2/proxy_v2_eval.txt
python exp/run_proxy_llm_compress_all_epoch.py --do_train=0 --choice_model=gpt4o > logs_v2/proxy_v2_compress_eval.txt

[2] continue run
python exp/run_proxy_llm_compress_icl_all_epoch.py > logs_v2/proxy_v2_compress_icl_2.txt

DEBUG!!!!
python exp/run_proxy_llm_compress_icl_all_epoch.py --do_train=0 --choice_model=gpt4o > logs_v2/proxy_v2_compress_icl_eval.txt 


Finish Ablation!

7.27
Run ape on all datasets
Run proxyv2+compress+icl on all datasets

[1] python exp/run_ape.py --num_epoch=5 --task=torchhub > logs_main/ape_torchhub.txt
[2] python exp/run_ape.py --num_epoch=5 --task=huggingface > logs_main/ape_huggingface.txt
[3] python exp/run_proxy_compress_icl1.py --num_epoch=2 --task=torchhub_icl1 > logs_main/proxy_v1_compress_icl1_torchhub.txt
[4] python exp/run_proxy_compress_icl1.py --num_epoch=2 --task=huggingface_icl1 > logs_main/proxy_v1_compress_icl1_huggingface.txt 
[5] python src/zeroshot_initialize.py --task=torchhub
[6] python src/zeroshot_initialize.py --task=huggingface
[7] python exp/run_proxy_compress_icl2.py --num_epoch=4 --task=torchhub_icl2 > logs_main/proxy_v1_compress_icl2_torchhub.txt
[8] python exp/run_proxy_compress_icl2.py --num_epoch=4 --task=huggingface_icl2 > logs_main/proxy_v1_compress_icl2_huggingface.txt

# select icl data for torchhub and huggingface
python src/icl_data.py --stage=1 --task=huggingface
python src/icl_data.py --stage=2 --task=huggingface
python src/icl_data.py --stage=1 --task=torchhub
python src/icl_data.py --stage=2 --task=torchhub

and copy the file from original to icl2

our method: proxy v1 compress on icl1 -> proxy v1 compress on icl2 -> 
python exp/run_proxy_compress_icl1.py --num_epoch=2 > logs_v2/proxy_v1_compress_icl1.txt 
python src/zeroshot_initialize.py
python exp/run_proxy_compress_icl2.py --num_epoch=4 > logs_v2/proxy_v1_compress_icl2.txt 
python exp/run_proxy_llm_compress_icl_all_epoch.py > logs_v2/proxy_v2_compress_icl_2.txt

7.28
Rerun ape on huggingface 
[1] python exp/run_ape.py --num_epoch=5 --task=huggingface > logs_main/ape_huggingface2.txt

Add proxy llm on icl2
[2] python exp/run_proxy_llm_compress_icl_all_epoch.py --task=huggingface_icl2 > logs_main/proxy_v2_compress_icl_huggingface.txt
[3] python exp/run_proxy_llm_compress_icl_all_epoch.py --task=torchhub_icl2 > logs_main/proxy_v2_compress_icl_torchhub.txt

Eval proxy llm on icl2
[4] python exp/run_proxy_llm_compress_icl_all_epoch.py --task=huggingface_icl2 --do_train=0 --choice_model=gpt4o > logs_main/proxy_v2_compress_icl_huggingface_eval.txt 
[5] python exp/run_proxy_llm_compress_icl_all_epoch.py --task=torchhub_icl2 --do_train=0 --choice_model=gpt4o > logs_main/proxy_v2_compress_icl_torchhub_eval.txt 

Other datasets
