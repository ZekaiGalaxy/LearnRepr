from .openai_public import get_embedding_query,get_embedding_context
from .utils import load_file, save_file

def get_data(args):
    if not args.task in ['lamp7']:
        APIs_description = load_file(f"data/{args.task}/APIs_description.json")
        APIs = load_file(f"data/{args.task}/APIs.json")
        Train_data = load_file(f"data/{args.task}/Train_data.json")
        Test_data = load_file(f"data/{args.task}/Test_data.json")
        API_train_data = load_file(f"data/{args.task}/API_train_data.json")
        API_test_data = load_file(f"data/{args.task}/API_test_data.json")
        return APIs_description, APIs, Train_data, Test_data, API_train_data, API_test_data
    elif args.task in ['lamp7']:
        data = load_file("data/lamp7/data_user.json")
        return data

def get_data_embedding(args,APIs,APIs_description,Train_data,Test_data):
    # API embed
    try:
        APIs_embed = load_file(f"embedding/{args.task}/APIs_embed_{args.embedding}.pt")
    except:
        APIs_embed = get_embedding_context(APIs, args)
        save_file(APIs_embed, f"embedding/{args.task}/APIs_embed_{args.embedding}.pt")

    # API description embed
    try:
        APIs_description_embed = load_file(f"embedding/{args.task}/APIs_description_embed_{args.embedding}.pt")
    except:
        APIs_description_embed={}
        for api_id, api in enumerate(APIs):
            APIs_description_embed[api]=get_embedding_context(APIs_description[api], args)
        save_file(APIs_description_embed, f"embedding/{args.task}/APIs_description_embed_{args.embedding}.pt")

    # Train data embed
    try:
        Train_data_embed = load_file(f"embedding/{args.task}/Train_data_embed_{args.embedding}.pt")
    except:
        Train_data_embed=[]
        for data in Train_data:
            data['prompt'] = get_embedding_query(data['prompt'], args)#.tolist()[0]
            Train_data_embed.append(data)
        save_file(Train_data_embed, f"embedding/{args.task}/Train_data_embed_{args.embedding}.pt")
    
    # Test data embed
    try:
        Test_data_embed = load_file(f"embedding/{args.task}/Test_data_embed_{args.embedding}.pt")
    except:
        Test_data_embed = []
        for data in Test_data:
            data['prompt'] = get_embedding_query(data['prompt'], args)#.tolist()[0]
            Test_data_embed.append(data)
        save_file(Test_data_embed, f"embedding/{args.task}/Test_data_embed_{args.embedding}.pt")

    # API train data embed
    API_train_data_embed = {}
    for api in APIs:
        API_train_data_embed[api] = []
        for data in Train_data_embed:
            if api.find('(') > 0:
                api = api[:api.find('(')]
            if api == data['completion'][0]:
                API_train_data_embed[api].append(data['prompt'])
    
    # API test data embed
    API_test_data_embed = {}
    for api in APIs:
        API_test_data_embed[api] = []
        for data in Test_data_embed:
            if api == data['completion'][0]:
                API_test_data_embed[api].append(data['prompt'])

    return APIs_embed,APIs_description_embed,Train_data_embed,Test_data_embed,API_train_data_embed,API_test_data_embed