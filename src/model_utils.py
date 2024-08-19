import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from .prompt import *
from .openai_public import *
from .utils import *

#################################################################################################
# Evaluation
#################################################################################################
def get_embedding_mean(api_embed):
	api_embeddings = {}
	apis = list(api_embed.keys())
	for api in apis:
		if len(api_embed[api]) > 1:
			api_embeddings[api] = get_mean(api_embed[api])
		else:
			api_embeddings[api] = api_embed[api][0]
	return api_embeddings

def llm_pred(api_desc, api_desc_embed, test_data, test_data_embed, args, record_prompt=False):
	apis = list(api_desc.keys())
	results = {}
	all_prompts = []
	all_responses = []
	for api_id, api in enumerate(apis):
		prompts = []
		for idx in range(len(test_data[api])):
			if args.choice_model == 'gpt4' or args.choice_model == 'gpt4o':
				prompt = f"""Which one is the most suitable API to complete the user's instruction?
{test_data[api][idx]}
You should refer to the patterns of the API and output 'I choose (Option [Letter])' as the answer."""
			else:
				prompt = f"""Which one is the most suitable API to complete the user's instruction?
ONLY output 'I choose (Option [Letter])' as the answer without any explanation.
{test_data[api][idx]}
"""
			for j in range(len(apis)):
				api_patterns = '\n'.join(pattern for pattern in api_desc[apis[j]])
				prompt += f"\n(Option {number_to_letter(j)}): API_name: {api_patterns}"
			prompts.append(prompt)
		all_prompts.extend(prompts)
		
		response = multi_query(prompts, args.choice_model)
		all_responses.extend(response)
		# preds = [x.replace('I choose (Option ', '').replace(')', '') for x in response]
		preds = [extract_choice(x) for x in response]
		preds = [letter_to_number(x) for x in preds]
		labels = [api_id] * len(test_data[api])
		results[api] = {
			'preds': preds,
			'labels': labels
		}
	if record_prompt and (args.choice_model == 'gpt4' or args.choice_model == 'gpt4o'):
		save_prompts(all_prompts, args)
		save_responses(all_responses, args)
	return results

def retriever_pred(api_desc, api_desc_embed, test_data, test_data_embed, args, record_prompt=False):
	api_embeddings = get_embedding_mean(api_desc_embed)
	api_matrix = torch.tensor([api_embeddings[api] for api in api_embeddings.keys()])
	apis = list(api_desc.keys())
	results = {}

	for api_id, api in enumerate(apis):
		try:
			instruct_embed = test_data_embed[api]
			instruct_matrix = torch.tensor(instruct_embed)
			preds = retrieve_topk(instruct_matrix, api_matrix, 1)
			preds = [x[0] for x in preds]
			labels = [api_id] * len(test_data[api])
		except: # no this class
			preds = []
			labels = []
		results[api] = {
			'preds': preds,
			'labels': labels
		}
	return results

def retriever_topk_llm_pred(api_desc, api_desc_embed, test_data, test_data_embed, args, record_prompt=False):
	k = args.retriever_topk
	api_embeddings = get_embedding_mean(api_desc_embed)
	api_matrix = torch.tensor(list(api_embeddings.values()))
	apis = list(api_desc.keys())
	results = {}
	all_prompts = []
	all_responses = []

	for api_id, api in enumerate(apis):
		prompts = []
		if len(test_data_embed[api]) == 0:
			results[api] = {
				'preds': [],
				'labels': []
			}
			continue
		# retriever topk
		test_embed = torch.tensor(test_data_embed[api])
		top_k_indices = retrieve_topk(test_embed, api_matrix, k) # [ins, k]


		for idx, top_k in enumerate(top_k_indices):
			if args.choice_model == 'gpt4' or args.choice_model == 'gpt4o':
				prompt = f"""Which one is the most suitable API to complete the user's instruction?
{test_data[api][idx]}
You should refer to the patterns of the API and output 'I choose (Option [Letter])' as the answer."""
			else:
				prompt = f"""Which one is the most suitable API to complete the user's instruction?
ONLY output 'I choose (Option [Letter])' as the answer without any explanation.
{test_data[api][idx]}
"""
			for i, index in enumerate(top_k):
				api_patterns = '\n'.join(pattern for pattern in api_desc[apis[index]])
				prompt += f"\n(Option {number_to_letter(i)}): API_name: {api_patterns}"
			prompts.append(prompt)
		all_prompts.extend(prompts)
		
		response = multi_query(prompts, args.choice_model)
		all_responses.extend(response)
		# preds = [x.replace('I choose (Option ', '').replace(')', '') for x in response]
		preds = [extract_choice(x) for x in response]
		new_preds = []
		for x in preds:
			try:
				new_preds.append(top_k[letter_to_number(x)])
			except:
				new_preds.append(-1)
		preds = new_preds
		labels = [api_id] * len(test_data[api])
		results[api] = {
			'preds': preds,
			'labels': labels
		}
	# breakpoint()
	# if record_prompt and args.choice_model == 'gpt4':
	if record_prompt:
		save_prompts(all_prompts, args)
		save_responses(all_responses, args)
	return results

def calculate_acc(results):
	accuracy = {}
	all_preds = []
	all_labels = []
	for api, result in results.items():
		preds = result['preds']
		labels = result['labels']
		all_preds.extend(preds)
		all_labels.extend(labels)
		if len(labels) == 0:
			accuracy[api] = -1
		else:
			accuracy[api] = sum(1 for pred, label in zip(preds, labels) if pred == label) / len(labels)
	accuracy['all'] = sum(1 for pred, label in zip(all_preds, all_labels) if pred == label) / len(all_labels)
	return accuracy

def get_wrong_samples(results, test_data, test_data_embed):
	fn_samples = {}
	fp_samples = {}
	fn_embeds = {}
	fp_embeds = {}
	apis = list(results.keys())
	for api in apis:
		fn_samples[api] = []
		fp_samples[api] = []
		fn_embeds[api] = []
		fp_embeds[api] = []
	
	for api in apis:
		preds = results[api]['preds']
		labels = results[api]['labels']
		for i, (pred, label) in enumerate(zip(preds, labels)):
			if pred != label:
				fn_samples[apis[label]].append(test_data[api][i])
				fn_embeds[apis[label]].append(test_data_embed[api][i])
				fp_samples[apis[pred]].append(test_data[api][i])
				fp_embeds[apis[pred]].append(test_data_embed[api][i])

	return fn_samples, fp_samples, fn_embeds, fp_embeds

def eval(api_desc, api_desc_embed, test_data, test_data_embed, args, mode='retriever', wrong_samples=False, record_prompt=False):
	# only record prompt when eval on train dataset
	if mode == 'retriever':
		results = retriever_pred(api_desc, api_desc_embed, test_data, test_data_embed, args, record_prompt=record_prompt)
	elif mode == 'llm':
		results = llm_pred(api_desc, api_desc_embed, test_data, test_data_embed, args, record_prompt=record_prompt)
	elif mode == 'retriever_llm':
		results = retriever_topk_llm_pred(api_desc, api_desc_embed, test_data, test_data_embed, args, record_prompt=record_prompt)
	else:
		raise ValueError('Invalid Mode!')
	accuracy = calculate_acc(results)

	if wrong_samples:
		fn_samples, fp_samples, fn_embeds, fp_embeds = get_wrong_samples(results, test_data, test_data_embed)
		return accuracy, fn_samples, fp_samples, fn_embeds, fp_embeds
	else:
		return accuracy

#################################################################################################
# Training
#################################################################################################
def repr_list_to_dict(repr_list, apis, name):
	repr_dict = {}
	for api_id, api in enumerate(apis):
		repr = repr_list[api_id]
		repr_dict[api] = repr
	return repr_dict

def extract_patterns(prompts, api_repr, name, args):
	print('Extracting Patterns...')
	drafts = multi_query(prompts, args.model)
	patterns = [get_patterns(draft) for draft in drafts]
	api_repr.extend(patterns)
	save_file(api_repr, name)
	return api_repr

def generate_api_repr(new_apis, old_api_desc, args):
	if args.zero_infer == 'icl':
		prompts = [icl_infer_prompt(api, old_api_desc) for api in new_apis]
	elif args.zero_infer == 'zeroshot':
		prompts = [zero_infer_prompt(api) for api in new_apis]
	new_api_desc = multi_query(prompts, args.model)

	return new_api_desc

#################################################################################################
# Compress
#################################################################################################
# def compress(api, patterns, patterns_embed, new_pattern, train_data_embed, args):
# 	mode = args.compress_mode
# 	first_pattern = patterns[0] # save the first key as name
# 	patterns = patterns[1:]
# 	patterns_embed = torch.tensor(patterns_embed[1:]) # [5,1536]

# 	# try:
# 	if mode == 'none':
# 		patterns += [new_pattern]
		
# 	elif mode == 'embedding_merge':
# 		new_pattern_embed = get_embedding_context(new_pattern, args)
# 		new_pattern_embed = torch.tensor(new_pattern_embed).unsqueeze(0)
# 		similarities = torch.matmul(F.normalize(new_pattern_embed), F.normalize(patterns_embed).t()) # [1,5]
# 		most_similar_index = torch.argmax(similarities).item() # 0
# 		compress_prompt = merge_two_patterns(api, patterns[most_similar_index], new_pattern)
# 		compressed_pattern = query(compress_prompt, args)
# 		compressed_pattern = get_patterns(compressed_pattern)
# 		assert len(compressed_pattern) == 1
# 		print(f'{mode} find {most_similar_index} to merge')
# 		patterns[most_similar_index] = compressed_pattern[0]

# 	elif mode == 'example_merge':
# 		new_pattern_embed = get_embedding_context(new_pattern, args)
# 		new_pattern_embed = torch.tensor(new_pattern_embed).unsqueeze(0)
# 		train_data_embed = torch.tensor(train_data_embed) # torch.Size([1793, 1536])
# 		pattern_samples = [[] for i in range(len(patterns)+1)]
# 		patterns_embed = torch.cat([patterns_embed, new_pattern_embed]) # [6,1536]
# 		similarities = torch.matmul(F.normalize(train_data_embed), F.normalize(patterns_embed).t()) # torch.Size([6, 1793])
# 		top_k_indices = torch.topk(similarities, k=3, dim=1).indices.tolist()
# 		for l, k_indices in enumerate(top_k_indices):
# 			for index in k_indices:
# 				pattern_samples[index].append(l)

# 		# find the most similar pattern in patterns using examples, and merge 2 patterns into 1
# 		new_pattern_samples = pattern_samples[-1]
# 		pattern_samples = pattern_samples[:-1]
# 		most_similar_index = -1
# 		max_score = -1
# 		for i, sample in enumerate(pattern_samples):
# 			score = len(set(sample) & set(new_pattern_samples)) / len(set(sample) | set(new_pattern_samples))
# 			if score > max_score:
# 				max_score = score
# 				most_similar_index = i
		
# 		print(f'{mode} find {most_similar_index} to merge')
# 		compress_prompt = merge_two_patterns(api, patterns[most_similar_index], new_pattern)
# 		compressed_pattern = query(compress_prompt, args)
# 		compressed_pattern = get_patterns(compressed_pattern)
# 		assert len(compressed_pattern) == 1
# 		patterns[most_similar_index] = compressed_pattern[0]

# 	elif mode == 'llm_merge':
# 		compress_prompt = merge_n_patterns(api, patterns, new_pattern)
# 		compressed_pattern = query(compress_prompt, args)
# 		compressed_pattern = get_patterns(compressed_pattern)
# 		assert len(compressed_pattern) <= len(patterns)
# 		patterns = compressed_pattern
# 	# except:
# 	# pass

# 	return [first_pattern] + patterns
def compress(api, patterns, patterns_embed, new_pattern, train_data_embed, args):
	first_pattern = patterns[0] # save the first key as name
	patterns = patterns[1:]
	patterns_embed = torch.tensor(patterns_embed[1:]) # [5,1536]

	new_pattern_embed = get_embedding_context(new_pattern, args)
	new_pattern_embed = torch.tensor(new_pattern_embed).unsqueeze(0)
	most_similar_index = retrieve_topk(new_pattern_embed, patterns_embed, 1)[0][0]

	compress_prompt = merge_two_patterns(api, patterns[most_similar_index], new_pattern)
	compressed_pattern = query(compress_prompt, args.model)
	compressed_pattern = get_patterns(compressed_pattern)
	save_prompts([compress_prompt], args)
	save_responses(compressed_pattern, args)
	if len(compressed_pattern) == 1:
		print(f'find {most_similar_index} to merge')
		patterns[most_similar_index] = compressed_pattern[0]
	else:
		print(f'failed to merge')

	return [first_pattern] + patterns

#################################################################################################
# Lamp7
#################################################################################################
def extract_user_pattern(text, args):
	print('Extracting User Pattern...')
	extract_prompt = user_extract_prompt(text)
	patterns = query(extract_prompt, args.model)
	patterns = get_patterns(patterns)
	return patterns

def eval_gen_train(user_profile, train, args):
	print('Eval Gen Train...')
	tests = [x[0] for x in train]
	labels = [x[1] for x in train]
	eval_prompt = [user_eval_prompt(user_profile, x) for x in tests]
	preds = multi_query(eval_prompt, args.model)
	mean_train_score, scores = calculate_gen_score(preds, labels)
	measures = [x['rouge-1'] for x in scores]

	if args.bad_strategy == 'threshold':
		bad_examples = [i for i, x in enumerate(measures) if x < args.bad_threshold]
	elif args.bad_strategy == 'percent':
		bad_examples = [x[0] for x in sorted(enumerate(measures), key=lambda x:x[1])[:int(len(measures)*args.bad_percent)]]
	elif args.bad_strategy == 'iqr':
		measures = sorted(measures)
		q1 = measures[int(len(measures)*0.4)]
		q3 = measures[int(len(measures)*0.6)]
		iqr = q3 - q1
		lower_bound = q1 - 0.25*iqr
		bad_examples = [i for i, x in enumerate(measures) if x < lower_bound]
	elif args.bad_strategy == 'fixed':
		measures = sorted(measures)
		bad_examples = [x[0] for x in measures][:args.bad_fixed]
	
	bad_examples = [[train[x][0], preds[x], train[x][1]] for x in bad_examples]
	print('#'*20+'Bad Examples:')
	for x in bad_examples:
		print(x)

	return mean_train_score, bad_examples

def eval_gen_test(user_profile, test, label, args):
	print('Eval Gen Test...')
	eval_prompt = user_eval_prompt(user_profile, test)
	pred = query(eval_prompt, args.model)
	return calculate_gen_score([pred], [label])[0]


