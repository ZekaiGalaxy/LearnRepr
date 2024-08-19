import sys
sys.path.append('/vepfs/wcf/G/zekai/root/.code/LearnRepr/src')
from utils import *

path = '/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v1_compress/prompts'
# ape
# print("input")
# print(calculate_token_cost_folder(path, dos=["fn","fp"],donts=["api5","api6","api7","api8","api9","response"])) 
# print("output")
# print(calculate_token_cost_folder(path, dos=["response"],donts=["api5","api6","api7","api8","api9"])) 

# proxy v1
# print("input")
# print(calculate_token_cost_folder(path, dos=["fn","fp"],donts=["5","6","7","8","9","response"])) 
# print("output")
# print(calculate_token_cost_folder(path, dos=["response"],donts=["5","6","7","8","9"])) 

# proxy v1 compress
# print("input")
# print(calculate_token_cost_folder(path, dos=["fn","fp"],donts=["5","6","7","8","9","response"])) 
# print("output")
# print(calculate_token_cost_folder(path, dos=["response"],donts=["5","6","7","8","9"])) 

# proxy v1 compress + icl
# print("input stage 1") # 7204
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl1_proxy_v1_compress_icl1/prompts", dos=["fn","fp","infer"],donts=["response"])) 
# print("output stage 1") # 5308
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl1_proxy_v1_compress_icl1/prompts", dos=["response"]))

# print("input stage 2") # 134784
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl2_proxy_v1_compress_icl2/prompts", dos=["fn","fp"],donts=["response"])) 
# print("output stage 2") # 28489
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl2_proxy_v1_compress_icl2/prompts", dos=["response"])) 

# proxy v2
# 163378+?
# 10069105 9170+27790
# print("input stage 1")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v1/prompts", dos=["fn","fp"],donts=["4","5","6","7","8","9","response"])) 
# # print("input stage 2 llama")
# # print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2/prompts", dos=["update","eval"],donts=["response"])) 
# print("input stage 2 gpt4o")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2/prompts", dos=["extract"],donts=["response"])) 
# print("output stage 1")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v1/prompts", dos=["response"],donts=["4","5","6","7","8","9"])) 
# # print("output stage 2 llama")
# # print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2/prompts", dos=["response"],donts=["extract"])) 
# print("output stage 2 gpt4o")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2/prompts", dos=["response"],donts=["update","eval"])) 

# proxy v2 compress
# 166528 + 45296
# 29246 + 10417
# print("input stage 1")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v1_compress/prompts", dos=["fn","fp"],donts=["4","5","6","7","8","9","response"])) 
# # print("input stage 2 llama")
# # print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2/prompts", dos=["update","eval"],donts=["response"])) 
# print("input stage 2 gpt4o")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2_compress/prompts", dos=["extract","compress"],donts=["response"])) 
# print("output stage 1")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v1_compress/prompts", dos=["response"],donts=["4","5","6","7","8","9"])) 
# # print("output stage 2 llama")
# # print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2/prompts", dos=["response"],donts=["extract"])) 
# print("output stage 2 gpt4o")
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_proxy_v2_compress/prompts", dos=["response"],donts=["update","eval"]))  

# proxy v2 compress icl
# print("input stage 1") # 7204
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl1_proxy_v1_compress_icl1/prompts", dos=["fn","fp","infer"],donts=["response"])) 
# print("output stage 1") # 5308
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl1_proxy_v1_compress_icl1/prompts", dos=["response"]))

# print("input stage 2") # 134784
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl2_proxy_v1_compress_icl2/prompts", dos=["fn","fp"],donts=["response"])) 
# print("output stage 2") # 28489
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl2_proxy_v1_compress_icl2/prompts", dos=["response"]))  

# print("input stage 3") # 155364
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl2_proxy_v2_compress_icl/prompts", dos=["fn","fp"],donts=["update","eval","response"])) 
# print("output stage 3") # 29122
# print(calculate_token_cost_folder("/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/tensorflow_icl2_proxy_v2_compress_icl/prompts", dos=["response"],donts=["update","eval"]))  