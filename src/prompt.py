import pdb
import random

def add_prompt(api, train_samples):
    return """Here is a retrieval task. The following samples is about the key: {0}. 
To help the user to distinguish the sample of this key with samples from other keys, you can extract essential and core patterns from samples of this key. The essential and core patterns always appear in the following samples and can represent the key. Each pattern should have a short and general description, together with 2~3 representative, short, and key cases.
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
For the cases, you must only copy the original words or phrases (nouns and verbs) from the samples as the cases. You must not directly copy the whole sample as your cases. Similar cases should be put together with a general form. 
You should generate less than 10 patterns, with less than 5 cases each pattern.
Samples:{1}""".format(api, train_samples)

def edit_element(patterns, fp_samples, most_similar_pattern):
    prompts = []
    count = {}
    for pid, pattern in enumerate(patterns):
        count[pid] = 0
        prompts.append(pattern + '\nWrong samples:')
    
    for sid, sample in enumerate(fp_samples):
        false_pattern = most_similar_pattern[sid][0]
        if count[false_pattern] < 6:
            prompts[false_pattern] += ' ' + sample
            count[false_pattern] += 1
    return prompts


def edit_prompt(api, patterns, fp_samples, most_similar_pattern, train_samples):
    prompts = edit_element(patterns, fp_samples, most_similar_pattern)
    return """Here is an edit task. The following patterns pertain to the key: {0}. 
However, some patterns mistakenly classify samples of other keys as belonging to {0}, which we refer to as 'wrong samples.' Your task is to modify the current patterns in order to reduce their similarity to the wrong samples, but these modified patterns should still accurately represent the key pattern for {0}.  You can refer to the correct samples of the key. 
To edit the pattern, you must re-write important words or phrase that are similar with the wrong samples, delete irrelevant words in the pattern for the key.  Additionaly, you can add a short description to claim the difference. You should try your best to make the pattern be different to the wrong samples. Patterns without wrong samples should not be edited. 
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
You should output the same number of patterns as before, with less than 5 cases each pattern. Each pattern should be the edited version of the corresponding previous pattern. 
Previous patterns and their wrong samples: 
{1}
Correct samples of key {0}: {2}""".format(api, prompts, train_samples[:10])

def merge_two_patterns(api, pattern1, pattern2):
    compress_prompt = """Here is a compression task. The following patterns pertain to the key: {0}.
You should compress the similar patterns into one pattern. The compressed pattern should still belong to the key. 
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
Similar patterns to be compressed:
{1}
{2}""".format(api, pattern1, pattern2)
    return compress_prompt

def merge_n_patterns(api, patterns, new_pattern):
    compress_prompt = """Here is a compression task. The following patterns pertain to the key: {0}.
You should change some of the existing patterns to cover the new pattern. The changed pattern should still belong to the key.
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
You should output the same number patterns as the existing patterns, with less than 5 cases each pattern.
Exisiting patterns to be changed:
{1}
New pattern:
{2}""".format(api, patterns, new_pattern)
    return compress_prompt

def user_update_prompt(user_profile, bad_examples):
    def join_samples(samples):
        sample_str = ""
        for sample in samples:
            sample_str += "Original sentence:\n" + sample[0] + "\nParaphrased sentence based on pattern:\n" + sample[1] + "\nLabel sentence in user style:\n" + sample[2] + "\n\n" 
        return sample_str
    return """Here is an edit task. The following patterns pertain to the user text style. 
However, some patterns lack sufficient representation or contain misleading information of the user's text style and may not be useful for paraphrasing sentences accordingly.  Your task is to modify the current patterns to make them more effective in guiding the paraphrasing of sentences to match the user's style.
To edit the pattern, you must re-write important words or phrase to make the pattern more representative of the user's text style. Additionaly, you can add a short description to describe the user's text style. You should try your best to make the pattern be more similar to the user's sentences.
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
You should output the same number of patterns as before, with less than 5 cases each pattern. Each pattern should be the edited version of the corresponding previous pattern. 
Previous patterns: 
{0}
The paraphrased result based on the previous patterns:
{1}""".format("\n".join(user_profile), join_samples(bad_examples))

def user_extract_prompt(text):
    return """Here is a retrieval task. The following are sentences said by a user. 
To distinguish the user text style from other styles, you can extract essential and core patterns from the user sentences. The essential and core patterns always appear in the following sentences and can represent the user's text style. Each pattern should have a short and general description, together with 2~3 representative, short, and key cases.
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
For the cases, you must only copy the original words or phrases (nouns and verbs) from the sentences as the cases. You must not directly copy the whole sentence as your cases. Similar cases should be put together with a general form.
You should generate less than 10 patterns, with less than 5 cases each pattern.
Sentences:{0}""".format(text)

def user_eval_prompt(user_profile, test_example):
    return """Paraphrase the following sentence based on the patterns of the user's text style.
User Style Pattern:
{0}
Sentences to be paraphrased:
{1}
You should output the paraphrased sentences without any explanation.""".format("\n".join(user_profile), test_example)

def icl_infer_prompt(api, old_api_descs):
    api_examples = ""
    for api, desc in old_api_descs.items():
        api_examples += "API: {0}\nDescription: {1}\n\n".format(api, desc)
    return """Here is a generation task. Your task is to generate patterns that represent the situations to call the api: {0}.
Each pattern should have a short and general description, together with 2~3 representative, short, and key instructions that belong to the api.
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
You should generate less than 10 patterns, with less than 5 cases each pattern.
Here are some examples:
{1}
""".format(api, api_examples)

def zero_infer_prompt(api):
    return """Here is a generation task. Your task is to generate patterns that represent the situations to call the api: {0}.
Each pattern should have a short and general description, together with 2~3 representative, short, and key instructions that belong to the api.
You should follow the following format, and no other explanation is needed:
Pattern: [Pattern Name]
  - Description: [Pattern Description]
  - Cases:
    - [Case 1]
    - ...
    - [Case N]
You should generate less than 10 patterns, with less than 5 cases each pattern.
""".format(api)

# def deletion_prompt(key,attribute_list):
# edited_lists = []
# for att_id, att in enumerate(attribute_list):
# edited_lists.append('Pattern {0}: '.format(att_id) + att)
# return """Here\'s a deletion task. The following patterns pertain to the key: {0}. However, some patterns may don’t belong to {0} and some patterns may be very similar. Your task is to delete the potential wrong pattern and merge the potential similar patterns into one pattern. The merged pattern should keep the details. And these modified patterns should still accurately represent the key patterns for {0}.  Patterns without wrong information should not be edited. Current patterns:{1}. Once you have made the necessary deletion and merge, you must output: ‘The patterns I want to delete:[Pattern number,Pattern number]’ and ‘The patterns I want to merge:[Pattern number,Pattern number], the merged pattern:[…]’ """.format(key,attribute_list)
# def add_prompt(key, previous_patterns,Training_samples):
# return "Here is a retrieval task. The following samples is about the key: {0}. To help the user to distinguish the sample of this key with samples from other keys, you can extract essential and core patterns from samples of this key. The essential and core patterns always appear in the following samples and can represent the key. Each pattern should have a short and general description, together with 2~3 representative, short, and key cases. For the cases, you must only copy the original words or phrases (nouns and verbs) from the samples as the cases. You must not directly copy the whole sample as your cases. You should generate new patterns that are different from previous patterns: {1}. Similar cases should be put together with a general form. Any two patterns should be exclusive. You need to find essential and core patterns as more as possible. You can list the patterns in a list. Samples:{2}".format(key, previous_patterns,Training_samples)
# def parse_attribute(key,sentence):
# return """parse this sentence into important chunks of the key : {0}. Sentence: {1}. Separate them with "" and put them into a python list. You must only output a python list.""".format(key,sentence)
# def self_update_prompt(key, Training_samples,previous_patterns):
# return 'Here is an intent classification task. Given you samples about the intent {0}. You should find new key patterns of the intent from the samples. The new patterns must have obvious difference to each of previous patterns {1}.  Similar cases should put into one new pattern with a general form. Any two new patterns should be exclusive. You cannot directly copy the whole sample as your response. You should find new patterns as more as possible. You can list the new patterns in a list. Training Samples:{1}.'.format(key, Training_samples,previous_patterns)
# def compress_prompt(key,patterns):
# return """You can compress the similar patterns into one pattern. The compressed pattern should still belong to the key: '{0}'. The compressed pattern should still follow the format of the similar patterns. You only need to output the compressed patterns. Similar patterns: {1}""".format(key,patterns)
# def distin_update_prompt(key, Training_samples,previous_patterns,others_patterns):
# return 'Here is an intent classification task. Previous patterns of intent {0} are similar to other intents. You can find new key patterns of the intent {0} that are different from each of previous patterns {2} and the patterns of other intents from the training samples {1} of the intent {0}. Similar cases should put into one new pattern with a general form. Any two new patterns should be exclusive. You cannot directly copy the whole sample as your response. You should find new patterns as more as possible. You can list the new patterns in a list.'.format(key,Training_samples,previous_patterns)
# def zero_shot_self(key):
# return 'Here is an intent classification task. The intent is {0}. You can find essential and core natural language patterns in sentence with this intent to help the user to distinguish the sentence with this intent with sentences with other intents. Essential and core patterns always appear in all sentences with the intent {0}. You can not just write a sentence with the intent {0} as your response. Similar patterns should be put into one pattern with a general form. Any two patterns should be exclusive. You need to find essential and core patterns as more as possible. You can list the patterns in a list.'.format(key)
# def upsup_self(key,unsup_samples):
# return 'Here is history samples with different intents.  You can find essential and core natural language patterns in both history samples with this intent and yourself knowledge to help the user to distinguish the sample with this intent with samples with other intents. Essential and core patterns always appear in all samples with the intent {0}. You can not just write a sentence with the intent {0} as your response. Similar patterns should be put into one pattern. Any two patterns should be exclusive. You need to find essential and core patterns as more as possible. You can list the patterns in a list. History sentence {1}'.format(key,unsup_samples)
# def clean_list(draft):
# return 'You can generate one python list that includes these pattern. One pattern represents one element of the list and surrounded by " ". You can not delete the content of the pattern. You must only output the one python list. Patterns:{0}'.format(draft)

