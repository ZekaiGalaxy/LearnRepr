import json
import torch
import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import os
import torch.nn.functional as F
import random
import tiktoken
from tqdm import tqdm
import shutil
import os
import re

random.seed(0)
torch.random.manual_seed(0)

def save_file(data, path):
    if path.endswith('.json'):
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    elif path.endswith('.pt'):
        torch.save(data, path)
    elif path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

def load_file(path):
    if path.endswith('.json'):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
    elif path.endswith('.pt'):
        data = torch.load(path)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif path.endswith('.txt'):
        with open(path, 'r') as f:
            data = f.read()
    return data

def myprint(x, name=None):
    if isinstance(x, list):
        print('*'*20)
        if name:
            print(name)
        for xx in x:
            myprint(xx)
            print('#'*20)
    elif isinstance(x, dict):
        print('*'*20)
        if name:
            print(name)
        for k, v in x.items():
            print(k)
            myprint(v)
            print('#'*20)
    elif isinstance(x, str):
        if name:
            print(name)
        print(x)

def get_patterns(pattern_str):
    ans = []
    patterns = pattern_str.split('Pattern')
    for p in patterns[1:]:
        ans.append('Pattern'+p.strip())
    return ans

def remove_str(string, sub):
    if isinstance(sub, list):
        for s in sub:
            string = string.replace(s, '')
    else:
        string = string.replace(sub, '')
    return string

def get_mean(embeddings):
    tensor_embeddings = torch.stack([torch.tensor(item) for item in embeddings])
    mean_embeddings = torch.mean(tensor_embeddings, dim=0)
    return mean_embeddings.tolist()

def remove_cases(pattern):
    pattern = pattern.split('- Cases')[0].strip()
    return pattern

def calculate_gen_score(preds, labels):
    # Assumes preds and labels are lists of strings (sentences)
    bleu_scores = {
        'bleu_1': [],
        'bleu_2': [],
        'bleu_3': [],
        'bleu_4': []
    }
    rouge = Rouge()
    scores_no_mean = []
    for pred, label in zip(preds, labels):
        reference = [label.split()] 
        candidate = pred.split()
        bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method1)
        bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        bleu1, bleu2, bleu3, bleu4 = round(bleu1*100, 2), round(bleu2*100, 2), round(bleu3*100, 2), round(bleu4*100, 2)
        bleu_scores['bleu_1'].append(bleu1)
        bleu_scores['bleu_2'].append(bleu2)
        bleu_scores['bleu_3'].append(bleu3)
        bleu_scores['bleu_4'].append(bleu4)
        scores = rouge.get_scores(pred, label)
        rouge1 = round(scores[0]['rouge-1']['f']*100,2)
        rouge2 = round(scores[0]['rouge-2']['f']*100,2)
        rougeL = round(scores[0]['rouge-l']['f']*100,2)

        scores_no_mean.append(
            {
                'bleu_1': bleu1,
                'bleu_2': bleu2,
                'bleu_3': bleu3,
                'bleu_4': bleu4,
                'rouge-1': rouge1,
                'rouge-2': rouge2,
                'rouge-l': rougeL
            }
        )
    
    avg_scores = {
        'bleu_1': round(sum(bleu_scores['bleu_1'])/len(bleu_scores['bleu_1']), 2),
        'bleu_2': round(sum(bleu_scores['bleu_2'])/len(bleu_scores['bleu_2']), 2),
        'bleu_3': round(sum(bleu_scores['bleu_3'])/len(bleu_scores['bleu_3']), 2),
        'bleu_4': round(sum(bleu_scores['bleu_4'])/len(bleu_scores['bleu_4']), 2),
        'rouge-1': round(sum([score['rouge-1'] for score in scores_no_mean])/len(scores_no_mean), 2),
        'rouge-2': round(sum([score['rouge-2'] for score in scores_no_mean])/len(scores_no_mean), 2),
        'rouge-l': round(sum([score['rouge-l'] for score in scores_no_mean])/len(scores_no_mean), 2)
    }
    
    return avg_scores, scores_no_mean

def peep(data, indent=0):
    def peep_print(*msg):
        print(" "*indent, *msg)
    if isinstance(data, dict):
        peep_print("Dict!")
        k = list(data.keys())[0]
        peep_print("Key:", k)
        peep(data[k], indent+4)
    elif isinstance(data, list):
        peep_print("List!")
        peep(data[0], indent+4)
    elif isinstance(data, str):
        peep_print(data[:100])
    elif isinstance(data, torch.Tensor):
        peep_print("Tensor!", data.shape)
    else:
        peep_print("Other!")

def save_prompt(prompt, name, args):
    path = os.path.join(args.exp_name, args.epoch, args.phase, name)
    with open(path, 'w') as f:
        f.write(prompt)

def number_to_letter(number):
    if 0 <= number <= 25:
        return chr(ord('A') + number)
    else:
        raise ValueError("Number must be between 1 and 26.")
    
def letter_to_number(letter):
    try:
        return ord(letter) - ord('A')
    except:
        return -1
    
# def retrieve_topk(embed1, embed2, k=1):
#     # breakpoint()
#     similarity = torch.matmul(F.normalize(embed1), F.normalize(embed2).t())
#     topk = torch.topk(similarity, k, dim=1)
#     return topk[1].tolist()

def retrieve_topk(embed1, embed2, k=1):
    embed1_norm = F.normalize(embed1, dim=1)
    embed2_norm = F.normalize(embed2, dim=1)
    similarity = torch.matmul(embed1_norm, embed2_norm.t())
    max_k = similarity.size(1)
    if k > max_k:
        k = max_k
    topk = torch.topk(similarity, k, dim=1)
    return topk[1].tolist()

def save_prompts(prompts, args):
    path = os.path.join(args.prompt_path,f"{args.prompt_name}.txt")
    with open(path, 'w') as f:
        if isinstance(prompts, list):
            for prompt in prompts:
                try:
                    f.write(prompt+'\n\n')
                except:
                    pass

        elif isinstance(prompts, str):
            try:
                f.write(prompts)
            except:
                pass

def save_responses(prompts, args):
    path = os.path.join(args.prompt_path,f"{args.prompt_name}_response.txt")
    with open(path, 'w') as f:
        if isinstance(prompts, list):
            for prompt in prompts:
                f.write(prompt+'\n\n')

        elif isinstance(prompts, str):
            f.write(prompts)

def debug_data(APIs_description, APIs, API_train_data, API_test_data, APIs_description_embed, API_train_data_embed, API_test_data_embed):
    k = 20
    APIs = APIs[:k]
    APIs_description = {api: APIs_description[api] for api in APIs}
    APIs_description_embed = {api: APIs_description_embed[api] for api in APIs}
    API_train_data = {api: API_train_data[api][:k] for api in APIs}
    API_test_data = {api: API_test_data[api][:k] for api in APIs}
    API_train_data_embed = {api: API_train_data_embed[api][:k] for api in APIs}
    API_test_data_embed = {api: API_test_data_embed[api][:k] for api in APIs}
    return APIs_description, APIs, API_train_data, API_test_data, APIs_description_embed, API_train_data_embed, API_test_data_embed

def random_sample(API_train_data, API_train_data_embed, n):
    instructions = []
    embeddings = []
    apis = list(API_train_data.keys())
    for api in API_train_data.keys():
        for instruct in API_train_data[api]:
            instructions.append((api,instruct))
        for embed in API_train_data_embed[api]:
            embeddings.append((api,embed))
    
    idx = torch.randperm(len(instructions))[:n]

    API_train_data = {api:[] for api in apis}
    API_train_data_embed = {api:[] for api in apis}

    for i in idx:
        api = instructions[i][0]
        API_train_data[api].append(instructions[i][1])
        API_train_data_embed[api].append(embeddings[i][1])

    return API_train_data, API_train_data_embed

def balanced_random_sample(API_train_data, API_train_data_embed, every_k):
    sampled_API_train_data = {api:[] for api in API_train_data.keys()}
    sampled_API_train_data_embed = {api:[] for api in API_train_data_embed.keys()}
    for api in API_train_data.keys():
        selected_idx = torch.randperm(len(API_train_data[api]))[:every_k]
        sampled_API_train_data[api] = [API_train_data[api][i] for i in selected_idx]
        sampled_API_train_data_embed[api] = [API_train_data_embed[api][i] for i in selected_idx]
    
    return sampled_API_train_data, sampled_API_train_data_embed

def process_null_text(text):
    if text == "":
        return " "
    return text

def get_token_len(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

def calculate_token_cost(path):
    text = load_file(path)
    return get_token_len(text)

def calculate_token_cost_folder(folder_path, dos=[], donts=[]):
    print(f'Calculating token cost for folder {folder_path}')
    files = os.listdir(folder_path)
    total_cost = 0
    for file in tqdm(files):
        is_valid = 0
        for x in dos:
            if x in file:
                is_valid = 1
        for x in donts:
            if x in file:
                is_valid = 0
        if not file.endswith('.txt'):
            is_valid = 0
        if is_valid == 0:
            continue
        else:
            print(file)
            total_cost += calculate_token_cost(f'{folder_path}/{file}')
    return total_cost

def copy_file(src, dst):
    source_folder = src
    destination_folder = dst
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # starts from 0-4
    pattern = re.compile(r'^[0-4]')
    for filename in os.listdir(source_folder):
        if pattern.match(filename):
            src_file = os.path.join(source_folder, filename)
            dest_file = os.path.join(destination_folder, filename)
            shutil.copy(src_file, dest_file)
            print(f"Copied: {src_file} to {dest_file}")

    print("Files copied successfully.")

def extract_choice(input_string):
    pattern = r'I choose \((Option \w+)\)|I choose (Option \w+)|I choose \((Option \w+)\)|I choose (Option \w+)|\((Option \w+)\)|(Option \w+)'
    match = re.search(pattern, input_string)
    if match:
        for group in match.groups():
            if group:
                return group.replace('Option ','')
    return None


##################################################################
if __name__ == '__main__':
#     pattern = """Pattern: Clustering and Grouping Analysis
# - Description: Discussions about grouping, clustering, or organizing texts based on similarity or related content.
# - Cases:
#     - group similar reviews together
#     - cluster similar sentences
#     - organize them into clusters
#     - categorize youtube channels
#     - Cluster articles"""
#     print(remove_cases(pattern))
    # copy_file('/data/zhangzk/Learn2Repr/output_v2/tensorflow_proxy_ss/prompts','/data/zhangzk/Learn2Repr/output_v2/tensorflow_proxy_sl/prompts')
    input_string1 = "I choose (Option A) because..."
    input_string2 = "I choose Option B) because..."
    input_string3 = "I choose (Option C) if...Option A"
    input_string4 = "I choose (Option B) Natural Language Processing Feature Extraction. I choose (Option D) Natural Language Processing Feature Extraction."
    input_string5 = "(Option E) is my choice..."
    input_string6 = "(Option F)"

    print(extract_choice(input_string1))  # Output: Option A
    print(extract_choice(input_string2))  # Output: Option B
    print(extract_choice(input_string3))  # Output: Option C
    print(extract_choice(input_string4))  # Output: Option D
    print(extract_choice(input_string5))  # Output: Option E
    print(extract_choice(input_string6))  # Output: Option F