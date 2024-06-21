import copy
import torch
import torch.nn.functional as F
import sys
sys.path.append(__file__.split('exp')[0])
from src.model_utils import *
from src.prompt import *
from src.data_process import *
from src.openai_public import *
from src.utils import *
import argparse
import os

# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='tensorflow', choices=['tensorflow','torchhub','huggingface','lamp7'])
parser.add_argument('--model', default='gpt4', type=str)
parser.add_argument('--embedding', default='gte-small', type=str)
parser.add_argument('--exp_name', default='ape_gte', type=str)
parser.add_argument('--num_epoch', default=10, type=int)
parser.add_argument('--do_train', default=1, type=int)
parser.add_argument('--do_eval', default=1, type=int)
parser.add_argument('--api', default=1, type=int)
parser.add_argument('--device', default=1, type=int)
parser.add_argument('--retriever_topk', default=10, type=int)
parser.add_argument('--output_dir', default='output', type=str)
args = parser.parse_args()



set_api(args)
set_model(args)
set_device(args)
os.makedirs(f'{args.output_dir}/{args.task}_{args.exp_name}', exist_ok=True)
os.makedirs(f'{args.output_dir}/{args.task}_{args.exp_name}/prompts', exist_ok=True)
args.prompt_path = f'{args.output_dir}/{args.task}_{args.exp_name}/prompts'

# Get data
APIs_description, APIs, Train_data, Test_data, API_train_data_ori, API_test_data = get_data(args)
APIs_embed, APIs_description_embed, Train_data_embed, Test_data_embed, API_train_data_embed_ori, API_test_data_embed = get_data_embedding(args, APIs, Train_data, Test_data)
print(f"get data done")


# APIs_description, APIs, API_train_data, API_test_data, APIs_description_embed, API_train_data_embed, API_test_data_embed = debug_data(APIs_description, APIs, API_train_data, API_test_data, APIs_description_embed, API_train_data_embed, API_test_data_embed)

if args.do_train:
    for epoch in range(args.num_epoch):
        # sample 5 for each api for val set every epoch
        API_train_data, API_train_data_embed = balanced_random_sample(API_train_data_ori, API_train_data_embed_ori, 5)
        for phase in ['fn', 'fp']: # fn: should yes but test no, fp: should no but test yes
            print(f"Starting Phase {phase}")
            args.epoch = epoch
            args.phase = phase
            
            # load from checkpoint, if ok skip this phase
            try:
                APIs_description = load_file(f'{args.output_dir}/{args.task}_{args.exp_name}/updated_{phase}_{epoch}.pt')
                APIs_description_embed = load_file(f'{args.output_dir}/{args.task}_{args.exp_name}/updated_embed_{phase}_{epoch}.pt')
                print(f'Load Checkpoint from {phase} {epoch} and Skip!')
                continue
            except:
                pass

            # Eval
            try:
                train_acc = load_file(f'{args.output_dir}/{args.task}_{args.exp_name}/train_acc_{phase}_{epoch}.json')
                fn_samples, fp_samples, fn_embeds, fp_embeds = load_file(f'{args.output_dir}/{args.task}_{args.exp_name}/wrong_samples_{phase}_{epoch}.pt')
            except:
                print(f"Eval train {phase}")
                args.prompt_name = f"{epoch}_{phase}_eval"
                train_acc, fn_samples, fp_samples, fn_embeds, fp_embeds = eval(APIs_description, APIs_description_embed, API_train_data, API_train_data_embed, args, mode='retriever_llm', wrong_samples=True, record_prompt=True)
                save_file(train_acc, f'{args.output_dir}/{args.task}_{args.exp_name}/train_acc_{phase}_{epoch}.json')
                save_file([fn_samples, fp_samples, fn_embeds, fp_embeds], f'{args.output_dir}/{args.task}_{args.exp_name}/wrong_samples_{phase}_{epoch}.pt')     

            # try:
            #     test_acc = load_file(f'{args.output_dir}/{args.task}_{args.exp_name}/test_acc_{phase}_{epoch}.json')
            # except:
            #     print(f"Eval test {phase}")
            #     test_acc = eval(APIs_description, APIs_description_embed, API_test_data, API_test_data_embed, args, mode='retriever_llm')
            #     save_file(test_acc, f'{args.output_dir}/{args.task}_{args.exp_name}/test_acc_{phase}_{epoch}.json')
            print(f"Epoch {epoch} Phase {phase} | Train acc {train_acc['all']}")

            # Get new patterns
            name = f'{args.output_dir}/{args.task}_{args.exp_name}/{phase}_{epoch}.pt'
            if os.path.exists(name):
                api_repr = load_file(name)
                print(f'Load Unupdated {phase} {epoch}!')
            else:
                api_repr = []

            false_samples = fn_samples if phase == 'fn' else fp_samples
            if phase == 'fp':
                false_apis = [api for api in false_samples.keys() if (len(APIs_description[api]) > 1 and len(false_samples[api]) > 0)]
            elif phase == 'fn':
                false_apis = [api for api in false_samples.keys() if len(false_samples[api]) > 0]
            prompts = []
            all_prompts = []

            for idx, api in enumerate(false_apis):
                # add prompt
                if phase == 'fn':
                    prompt = add_prompt(api, fn_samples[api]).replace('Instruction: ', '')
                    all_prompts.append(prompt)
                    if idx < len(api_repr):
                        continue
                    prompts.append(prompt)

                elif phase == 'fp':
                    if len(APIs_description[api]) > 1:
                        instruction_embeds = torch.tensor(fp_embeds[api])
                        pattern_embeds = torch.tensor(APIs_description_embed[api][1:])   
                        most_similar_pattern = retrieve_topk(instruction_embeds, pattern_embeds, 1)

                        prompt = edit_prompt(api, APIs_description[api][1:], fp_samples[api], most_similar_pattern, API_train_data[api])
                        prompt = prompt.replace('Instruction: ', '')
                        all_prompts.append(prompt)
                        if idx < len(api_repr):
                            continue
                        prompts.append(prompt)

                # execute prompt
                if len(prompts) == 64 or idx == len(false_apis) - 1:
                    api_repr = extract_patterns(prompts, api_repr, name, args)
                    prompts = []

            args.prompt_name = f"{epoch}_{phase}_extract"
            save_prompts(all_prompts, args)
            save_responses(["\n".join(x) for x in api_repr], args)
            api_repr = repr_list_to_dict(api_repr, false_apis, name) # list to dict

            # If boosts performance, then update
            for api_id, api in enumerate(api_repr.keys()):
                if phase == 'fp' and not (len(api_repr[api]) == len(APIs_description[api]) - 1):
                    print(f'Failed in fp {api_id} {api}! previous patterns {len(APIs_description[api]) - 1} current patterns {len(api_repr[api])}!')
                    continue

                for pid, pattern in enumerate(api_repr[api]):
                    args.prompt_name = f"{epoch}_{phase}_api{api_id}_pattern{pid}_update"
                    if phase == 'fp':
                        pseudo_description = copy.deepcopy(APIs_description)
                        pseudo_description[api][pid+1] = pattern
                        pseudo_embed = copy.deepcopy(APIs_description_embed)
                        pseudo_embed[api][pid+1] = get_embedding_context(pattern, args)

                        pseudo_acc = eval(pseudo_description, pseudo_embed, API_train_data, API_train_data_embed, args, mode='retriever_llm', record_prompt=True)
                        
                        if pseudo_acc['all'] > train_acc['all']:
                            APIs_description = pseudo_description
                            APIs_description_embed = pseudo_embed
                            print(f"{phase} {api} {pid} Train Acc: {train_acc['all']} -> {pseudo_acc['all']}! [Edited]")
                            train_acc = pseudo_acc
                        else:
                            print(f"{phase} {api} {pid} Train Acc: {train_acc['all']} -> {pseudo_acc['all']}! [Ignored]")
                    
                    elif phase == 'fn':
                        pseudo_description = copy.deepcopy(APIs_description)
                        pseudo_description[api].append(pattern)
                        pseudo_embed = copy.deepcopy(APIs_description_embed)
                        pseudo_embed[api].append(get_embedding_context(pattern, args))
                        
                        pseudo_acc = eval(pseudo_description, pseudo_embed, API_train_data, API_train_data_embed, args, mode='retriever_llm', record_prompt=True)
                        
                        if pseudo_acc['all'] > train_acc['all']:
                            APIs_description = pseudo_description
                            APIs_description_embed = pseudo_embed
                            print(f"{phase} {api} {pid} Train Acc: {train_acc['all']} -> {pseudo_acc['all']}! [Added], Pattern Num: {len(APIs_description[api])}")
                            train_acc = pseudo_acc
                        else:
                            print(f"{phase} {api} {pid} Train Acc: {train_acc['all']} -> {pseudo_acc['all']}! [Ignored], Pattern Num: {len(APIs_description[api])}")
            
            # save
            save_file(APIs_description, f'{args.output_dir}/{args.task}_{args.exp_name}/updated_{phase}_{epoch}.pt')
            save_file(APIs_description_embed, f'{args.output_dir}/{args.task}_{args.exp_name}/updated_embed_{phase}_{epoch}.pt')

if args.do_eval:
    APIs_description = load_file(f'{args.output_dir}/{args.task}_{args.exp_name}/updated_fp_{args.epoch-1}.pt')
    APIs_description_embed = load_file(f'{args.output_dir}/{args.task}_{args.exp_name}/updated_embed_fp_{args.epoch-1}.pt')
    print('Load API Desc and Embed!')            

    # Final Test
    print("Final Test")
    train_acc = eval(APIs_description, APIs_description_embed, API_train_data, API_train_data_embed, args, mode='retriever_llm')
    print(f"Train Acc: {train_acc}")
    test_acc = eval(APIs_description, APIs_description_embed, API_test_data, API_test_data_embed, args, mode='retriever_llm')
    print(f"Test Acc: {test_acc}")