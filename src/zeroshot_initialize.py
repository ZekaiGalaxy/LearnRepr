from utils import *
import argparse
import os
from prompt import *
from openai_public import *
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='tensorflow')
parser.add_argument('--embedding', default='gte-small')
args = parser.parse_args()
args.prompt_path = f"/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/{args.task}_icl1_proxy_v1_compress_icl1/prompts"

# load three patterns
x = load_file(f'/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/{args.task}_icl1_proxy_v1_compress_icl1/updated_fp_1.pt')
print(x)
all_apis = load_file(f'/vepfs/wcf/G/zekai/root/.code/LearnRepr/data/{args.task}_icl2/APIs.json')

prompts = ""
def query_prompt(api):
    global prompts
    prompt = zero_infer_prompt(api, x)
    answer = query1(prompt)
    prompts += prompt + "\n"
    args.prompt_name = f"zeroshot_{api}"
    save_responses(answer, args)
    return answer
for api in all_apis:
    query_prompt(api)
args.prompt_name = f"zeroshot_infer"
save_prompts(prompts, args)

def process_answer(api, answer):
    patterns = answer.split("Pattern:")[1:]
    patterns = ["Pattern:"+x for x in patterns]
    return [api]+patterns[:3]

y = {}

for api in all_apis:
    if api in x:
        y[api] = x[api]
        continue
    answer = load_file(f"/vepfs/wcf/G/zekai/root/.code/LearnRepr/output_v2/{args.task}_icl1_proxy_v1_compress_icl1/prompts/zeroshot_{api}_response.txt")
    # print(api)
    api_pattern = process_answer(api, answer)
    y[api] = api_pattern

save_file(y, f"/vepfs/wcf/G/zekai/root/.code/LearnRepr/data/{args.task}_icl2/APIs_description.json")


# for x in patterns:
#     print(x)

# def save_prompt(prompt, name, args):
#     path = os.path.join(args.exp_name, args.epoch, args.phase, name)
#     with open(path, 'w') as f:
#         f.write(prompt)

# def save_responses(prompts, args):
#     path = os.path.join(args.prompt_path,f"{args.prompt_name}_response.txt")
#     with open(path, 'w') as f:
#         if isinstance(prompts, list):
#             for prompt in prompts:
#                 f.write(prompt+'\n\n')

#         elif isinstance(prompts, str):
#             f.write(prompts)


#             save_prompts(all_prompts, args)
#             save_responses(["\n".join(x) for x in api_repr], args)
# save_file(APIs_description_embed, f'{args.output_dir}/{args.task}_{args.exp_name}/updated_embed_{phase}_{epoch}.pt')

# generate for other patterns
    

# save