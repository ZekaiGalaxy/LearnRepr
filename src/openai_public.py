from openai import AzureOpenAI
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import backoff 
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer
import torch
import time
import warnings
import sys
sys.path.append("/vepfs/wcf/G/zekai/root/.code/LearnRepr/src")
from utils import *
warnings.filterwarnings("ignore")
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

load_dotenv()
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential,"https://cognitiveservices.azure.com/.default")
client = AzureOpenAI(
  azure_endpoint="https://nllearn4o0.openai.azure.com/",
  azure_ad_token_provider=token_provider,
  api_version="2024-02-15-preview",
  # api_version="2024-02-01",
  max_retries=5,
)

llama_clients = [
    OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{8000+i}/v1"
    )
    for i in range(1,8)
]


global client1, client2
global api
global llama3, llama3_tokenizer
global tokenizer, embedding_model
global device

api = 1
device = 'cuda:0'

def set_model(args):
    global llama3, llama3_tokenizer, tokenizer, embedding_model, device
    llama3, llama3_tokenizer, tokenizer, embedding_model = None, None, None, None
    if args.model == 'llama3': # no need for choice model
        llama3_path = "/vepfs/wcf/G/zekai/models/llama3"
        llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_path, padding_side="left")
        llama3 = AutoModelForCausalLM.from_pretrained(llama3_path, torch_dtype=torch.bfloat16).to(device)
    if args.embedding == 'gte-small':
        path = 'gte-small'
        path = '/vepfs/wcf/G/zekai/models/gte-small'
        tokenizer = AutoTokenizer.from_pretrained(path)
        embedding_model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device)
    elif args.embedding == 'gte-large':
        raise ValueError("Not implemented")
    elif args.embedding == 'sfr':
        raise ValueError("Not implemented")
    elif args.embedding == 'e5':
        raise ValueError("Not implemented")
    elif args.embedding == 'phi3':
        raise ValueError("Not implemented")

# client1 = AzureOpenAI(
#     api_key="54b5441e258049ad9291387d8db50372",  
#     api_version="2024-02-01",
#     azure_endpoint=f"https://learnnl6.openai.azure.com/"
# )
# client2 = AzureOpenAI(
#     api_key="ff2c0f1f6cdd480e81489ba52f094b60",  
#     api_version="2024-02-01",
#     azure_endpoint=f"https://learnnl2.openai.azure.com/"
# )
    
client1 = client
client2 = client

def set_api(args):
    global api
    api = args.api

def set_device(args):
    global device
    device = f'cuda:{args.device}'

def get_detailed_instruct(query):
    task = 'Given a user instruction, retrieve relevant api name that answer the instruction.'
    query = query.replace('Instruction: ','')
    return f'Instruct: {task}\nQuery: {query}'

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embedding_context(text, args):
    if args.embedding == "gte-small":
        if isinstance(text, list):
            batch_dict = tokenizer(text, padding=True,return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0].to('cpu').tolist()

        else:
            batch_dict = tokenizer([text], padding=True,return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = outputs.last_hidden_state[0, 0].to('cpu').tolist()

        return embeddings
    elif args.embedding == "gte-large" or args.embedding == "sfr" or args.embedding == "e5":
        if isinstance(text, list):
            text = [process_null_text(x) for x in text]
            batch_dict = tokenizer(text, padding=True, return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to('cpu').tolist()
        else:
            text = process_null_text(text)
            batch_dict = tokenizer([text], padding=True, return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0].to('cpu').tolist()
        return embeddings
    elif args.embedding == 'phi3':
        if isinstance(text, list):
            inputs = tokenizer(text, padding=True, max_length=2048, truncation=True, return_tensors='pt')
            inputs = {k: v.to(embedding_model.device) for k, v in inputs.items()}
            outputs = embedding_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1][:, -1, :] 
            return last_hidden_state.to('cpu').tolist()
        else:
            inputs = tokenizer([text], padding=True, max_length=2048, truncation=True, return_tensors='pt')
            inputs = {k: v.to(embedding_model.device) for k, v in inputs.items()}
            outputs = embedding_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1][:, -1, :] 
            return last_hidden_state[0].to('cpu').tolist()
    else:
        if isinstance(text, list):
            embeddings = client1.embeddings.create(input = text, model="text-embedding-ada-002").data
            return [embedding.embedding for embedding in embeddings]
        elif isinstance(text, str):
            return client1.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding
    
def get_embedding_query(text, args):
    if args.embedding == "gte-small":
        if isinstance(text, list):
            batch_dict = tokenizer(text, padding=True,return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0].to('cpu').tolist()

        else:
            batch_dict = tokenizer([text], padding=True,return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = outputs.last_hidden_state[0, 0].to('cpu').tolist()

        return embeddings
    elif args.embedding == "gte-large" or args.embedding == "sfr" or args.embedding == "e5":
        if isinstance(text, list):
            text = [process_null_text(x) for x in text]
            text = [get_detailed_instruct(x) for x in text]
            batch_dict = tokenizer(text, padding=True, return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to('cpu').tolist()

        else:
            text = process_null_text(text)
            text = get_detailed_instruct(text)
            batch_dict = tokenizer([text], padding=True, return_tensors='pt')
            batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}
            outputs = embedding_model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0].to('cpu').tolist()

        return embeddings
    elif args.embedding == 'phi3':
        if isinstance(text, list):
            inputs = tokenizer(text, padding=True, max_length=2048, truncation=True, return_tensors='pt')
            inputs = {k: v.to(embedding_model.device) for k, v in inputs.items()}
            outputs = embedding_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1][:, -1, :] 
            return last_hidden_state.to('cpu').tolist()
        else:
            inputs = tokenizer([text], padding=True, max_length=2048, truncation=True, return_tensors='pt')
            inputs = {k: v.to(embedding_model.device) for k, v in inputs.items()}
            outputs = embedding_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1][:, -1, :] 
            return last_hidden_state[0].to('cpu').tolist()
    else:
        if isinstance(text, list):
            return client1.embeddings.create(input = text, model="text-embedding-ada-002").data[0].embedding
        elif isinstance(text, str):
            return client1.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding

def query_azure_openai_chatgpt_chato(client, query, temperature=0.0):
    try:
        completion = completions_with_backoff(
            client=client,
            model="gpt-4o", # "gpt-4": "2024-02-01"
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
        )
    except:
        return ""
    return completion.choices[0].message.content

def query_azure_openai_chatgpt_chat(client, query, temperature=0.0):
    try:
        completion = completions_with_backoff(
            client=client,
            model="gpt-4", # "gpt-4": "2024-02-01"
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query},
            ],
            temperature=temperature,
        )
    except:
        return ""
    return completion.choices[0].message.content

def chat_llama3(params):
    query, i = params
    client = llama_clients[i]
    # try:
    chat_response = client.chat.completions.create(
        model="/vepfs/wcf/G/zekai/models/llama3",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        temperature=0.0
    )
    return chat_response.choices[0].message.content
    # except:
    #     print(f"Error with query: {query}")
    #     return ""



def query_llama3(query):
    if isinstance(query, list):
        # try:
        qs = [{"role": "user", "content": q} for q in query]
        texts = [llama3_tokenizer.apply_chat_template([x], add_generation_prompt=True, tokenize=False) for x in qs]
        llama3_tokenizer.pad_token_id = llama3_tokenizer.eos_token_id  # Set a padding token
        inputs = llama3_tokenizer(texts, padding="longest", return_tensors="pt")
        inputs = {key: val.to(llama3.device) for key, val in inputs.items()}

        outputs = llama3.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

        # Decode the responses
        responses = []
        for i in range(outputs.size(0)):
            response = outputs[i][inputs['input_ids'].shape[-1]:]
            response = llama3_tokenizer.decode(response, skip_special_tokens=True)
            responses.append(response)
        return responses
        # except Exception as e:
        #     print(e)
        #     return ["" for _ in query]
    else:
        try:
            messages = [
                {"role": "user", "content": query},
            ]

            input_ids = llama3_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(llama3.device)

            terminators = [
                llama3_tokenizer.eos_token_id,
                llama3_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = llama3.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=False
            )
            response = outputs[0][input_ids.shape[-1]:]
            response = llama3_tokenizer.decode(response, skip_special_tokens=True)
        except Exception as e:
            print(e)
            response = ""
        return response

def query1(query):
    ans = query_azure_openai_chatgpt_chat(client1, query)
    # print(ans, flush=True)  
    return ans
def query2(query):
    ans = query_azure_openai_chatgpt_chat(client2, query)
    return ans
def query1o(query):
    ans = query_azure_openai_chatgpt_chato(client1, query)
    # print(ans, flush=True)  
    return ans
def query2o(query):
    ans = query_azure_openai_chatgpt_chato(client2, query)
    return ans
def query(query, model):
    if model == "gpt4":
        if api == 1:
            return query1(query)
        else:
            return query2(query)
    elif model == "gpt4o":
        if api == 1:
            return query1o(query)
        else:
            return query2o(query)
    elif model == "llama3":
        return chat_llama3(query,0)
    else:
        raise ValueError("Invalid model name")


def batch_queries(queries, batch_size):
    for i in range(0, len(queries), batch_size):
        yield queries[i:i + batch_size]

def multi_query(queries, model):
    if model == "gpt4o":
        # global api
        outputs = [None] * len(queries)  # Initialize a list to store the results in order
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            len_queries = len(queries)
            
            if api == 1:
                futures1 = {executor.submit(query1o, query): i for i, query in enumerate(queries)}
            else:
                futures1 = {executor.submit(query2o, query): i for i, query in enumerate(queries)}
            # futures2 = {executor.submit(query2, query): i + len_queries//2 for i, query in enumerate(queries[len_queries//2:])}
            # all_futures = {**futures1, **futures2}
            for future in tqdm(concurrent.futures.as_completed(futures1), total=len_queries, desc="Processing Queries"):
                outputs[futures1[future]] = future.result()
        return outputs

    elif model == "gpt4":
        # global api
        outputs = [None] * len(queries)  # Initialize a list to store the results in order
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            len_queries = len(queries)
            
            if api == 1:
                futures1 = {executor.submit(query1, query): i for i, query in enumerate(queries)}
            else:
                futures1 = {executor.submit(query2, query): i for i, query in enumerate(queries)}
            # futures2 = {executor.submit(query2, query): i + len_queries//2 for i, query in enumerate(queries[len_queries//2:])}
            # all_futures = {**futures1, **futures2}
            for future in tqdm(concurrent.futures.as_completed(futures1), total=len_queries, desc="Processing Queries"):
                outputs[futures1[future]] = future.result()
        return outputs

    elif model == "llama3":
        chunk_size = len(queries) // len(llama_clients)
        params = [(queries[i*chunk_size + j], i) for i in range(len(llama_clients)) for j in range(chunk_size)]
        outputs = [None] * len(params)  # Initialize a list to store the results in order
        with concurrent.futures.ThreadPoolExecutor(max_workers=800) as executor:
            futures = {executor.submit(chat_llama3, param): idx for idx, param in enumerate(params)}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Queries"):
                idx = futures[future]
                outputs[idx] = future.result()

        return outputs
        # outputs = []
        # batch_size = 8
        # for batch in tqdm(batch_queries(queries, batch_size), total=len(queries)//batch_size, desc="Processing Queries"):
        #     outputs.extend(query_llama3(batch))
        # return outputs
    else:
        raise ValueError("Invalid model name")


if __name__ =='__main__':
    queries = ["How to make a cake, give me the recipe"]*160
    print(multi_query(queries, "llama3"))
    # print(query1("hello!"))
    # for k,v in API_dic.items():
    #     print(k,v)
    #     client = AzureOpenAI(
    #         api_key=v,  
    #         api_version="2023-11-06-preview", #"2024-02-01", 
    #         azure_endpoint=f"https://{k}.openai.azure.com/"
    #     )

    #     try:
    #         print(query_azure_openai_chatgpt_chat("Hello, how are you?"))
    #     except Exception as e:
    #         print(e)
    
    # test = ["Hello, how are you?", "I am fine, thank you.", "What are you doing?", "I am working on a project."]
    # x = multi_threading_running(get_embedding_context, test)
    # y = get_embedding_context(test)

    # print(type(x))
    # print(type(y))
    # print(type(x[0]))
    # print(type(y[0]))

    # assert x[0] == y[0]
    # assert x[1] == y[1]
    # assert x[2] == y[2]
    # assert x[3] == y[3]

    # x = get_embedding_context("Hello, world!")
    # print(torch.tensor(x).shape)

    # import time 
    # t1 = time.time()
    # queries = [f"Please output {i}" for i in range(20)]
    # outputs = multi_query(queries)
    # for output in outputs:
    #     print(output)
    # print('Time:', time.time()-t1)

    # outputs = [query1(query) for query in queries]
    # for output in outputs:
    #     print(output)
    # print('Time:', time.time()-t1)


# API_dic = {
#     'gcrgpt4aoai5': '653880d85b6e4a209206c263d7c3cc7a',
#     'yaobo': 'c1dc6a15023d466aa66afcf1c2fdcbb2',
#     'yaobo2': '75c420c623a84d77b6071671819d2d52',
#     'yaobo4': '61d5ba81490d4640b72cda891b5a98b8',
#     'yaobo5': '33bc9e9477574cd5b378f881c04e2f55',
#     'leiji': '5f0662c218414ad6a9069fb0eddf5e45',
#     "leiji2": "e8e2b2c8680148e7acef21c36c80b05e",
#     "leiji3": "0d5bed7b43324a88b22d37aa455fcd62",
#     "leiji4":"0de68f5001214c0da6f70a13328ff8a6",
#     "leiji5": "327415cdff7f4385b76d5ded0a8a1653",
#     'nlc2': '804c27272f5049ffb1d1668930201997',
#     'nlc3': '9135f9e5b2cc49ae8222b111dafbc4f2',
#     'nlc4': '00e3ad8962984684a3fd97890545fe7c',
#     'nlc5': '65098fbc406f4f1c96b0b3a7677b04d1',
#     'nlc6': 'e0e169be92fd41c2add67b3f4873cae3',
#     "learnnl2": "ff2c0f1f6cdd480e81489ba52f094b60",
#     "learnnl6": "54b5441e258049ad9291387d8db50372"
# }