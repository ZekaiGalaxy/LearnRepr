# load
# xxx -> xxx_old
# stage1: normal train, task=xxx_old
# stage2: icl, load examples, design prompt and test
# stage2 baseline: no load examples, design prompt and test

from utils import *
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='tensorflow')
args = parser.parse_args()

API_test_data = load_file(f"data/{args.task}/API_test_data.json")
API_train_data = load_file(f"data/{args.task}/API_train_data.json")
APIs = load_file(f"data/{args.task}/APIs.json")
APIs_description = load_file(f"data/{args.task}/APIs_description.json")
Train_data = load_file(f"data/{args.task}/Train_data.json")
Test_data = load_file(f"data/{args.task}/Test_data.json")

APIs_embed = load_file(f"embedding/{args.task}/APIs_embed.pt")
APIs_description_embed = load_file(f"embedding/{args.task}/APIs_description_embed.pt")
Train_data_embed = load_file(f"embedding/{args.task}/Train_data_embed.pt")
Test_data_embed = load_file(f"embedding/{args.task}/Test_data_embed.pt")

# problem1: given task, which API to select? - all the apis, train num and test num
# print(f"Total {len(APIs)} APIs")
# for api in APIs:
#     train_num = len(API_train_data[api])
#     test_num = len(API_test_data[api])
#     print(f"train {train_num} test {test_num} {api}")

# problem2: given task and selected api, prepare all the old data and the new data 
# prepare old data
selected_APIs = APIs[:30]

selected_idx = [i for i in range(len(APIs)) if APIs[i] in selected_APIs]
old_API_test_data = {k:v for k,v in API_test_data.items() if k in selected_APIs}
old_API_train_data = {k:v for k,v in API_train_data.items() if k in selected_APIs}
old_APIs = selected_APIs
old_APIs_description = {k:v for k,v in APIs_description.items() if k in selected_APIs}
old_Train_data = [data for data in Train_data if data['completion'][0] in selected_APIs]
old_Test_data = [data for data in Test_data if data['completion'][0] in selected_APIs]
old_APIs_embed = [x[idx] for idx,x in enumerate(APIs_embed) if idx in selected_idx]
old_APIs_description_embed = {k:v for k,v in APIs_description_embed.items() if k in selected_APIs}
old_Train_data_embed = [data for data in Train_data_embed if data['completion'][0] in selected_APIs]
old_Test_data_embed = [data for data in Test_data_embed if data['completion'][0] in selected_APIs]

save_file(old_API_test_data, f"data/{args.task}_old/API_test_data.json")
save_file(old_API_train_data, f"data/{args.task}_old/API_train_data.json")
save_file(old_APIs, f"data/{args.task}_old/APIs.json")
save_file(old_APIs_description, f"data/{args.task}_old/APIs_description.json")
save_file(old_Train_data, f"data/{args.task}_old/Train_data.json")
save_file(old_Test_data, f"data/{args.task}_old/Test_data.json")
save_file(old_APIs_embed, f"embedding/{args.task}_old/APIs_embed.pt")
save_file(old_APIs_description_embed, f"embedding/{args.task}_old/APIs_description_embed.pt")
save_file(old_Train_data_embed, f"embedding/{args.task}_old/Train_data_embed.pt")
save_file(old_Test_data_embed, f"embedding/{args.task}_old/Test_data_embed.pt")

# prepare new data
new_APIs = [x for x in APIs if x not in selected_APIs]
new_idx = [i for i in range(len(APIs)) if APIs[i] in new_APIs]
new_API_test_data = {k:v for k,v in API_test_data.items() if k in new_APIs}
new_API_train_data = {k:v for k,v in API_train_data.items() if k in new_APIs}
new_APIs_description = {k:v for k,v in APIs_description.items() if k in new_APIs}
new_Train_data = [data for data in Train_data if data['completion'][0] in new_APIs]
new_Test_data = [data for data in Test_data if data['completion'][0] in new_APIs]
new_APIs_embed = [x[idx] for idx,x in enumerate(APIs_embed) if idx in new_idx]
new_APIs_description_embed = {k:v for k,v in APIs_description_embed.items() if k in new_APIs}
new_Train_data_embed = [data for data in Train_data_embed if data['completion'][0] in new_APIs]
new_Test_data_embed = [data for data in Test_data_embed if data['completion'][0] in new_APIs]

save_file(new_API_test_data, f"data/{args.task}_new/API_test_data.json")
save_file(new_API_train_data, f"data/{args.task}_new/API_train_data.json")
save_file(new_APIs, f"data/{args.task}_new/APIs.json")
save_file(new_APIs_description, f"data/{args.task}_new/APIs_description.json")
save_file(new_Train_data, f"data/{args.task}_new/Train_data.json")
save_file(new_Test_data, f"data/{args.task}_new/Test_data.json")
save_file(new_APIs_embed, f"embedding/{args.task}_new/APIs_embed.pt")
save_file(new_APIs_description_embed, f"embedding/{args.task}_new/APIs_description_embed.pt")
save_file(new_Train_data_embed, f"embedding/{args.task}_new/Train_data_embed.pt")
save_file(new_Test_data_embed, f"embedding/{args.task}_new/Test_data_embed.pt")



# for the new data, we just need to (1) generate descriptions for each api (2) test

