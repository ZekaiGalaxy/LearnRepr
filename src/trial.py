import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

device = 'cuda:1'

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    query = query.replace('Instruction: ','')
    return f'Instruct: {task_description}\nQuery: {query}'

# def get_detailed_instruct(task_description: str, query: str) -> str:
#     return f'{query}'

task = 'Given a user instruction, retrieve relevant api name that answer the instruction.'
queries = [
    get_detailed_instruct(task, "Instruction: I'm planning to do sentiment analysis on a bunch of news articles. Help me convert the article content to 20-dimensional vectors."),
    get_detailed_instruct(task, "Instruction: Analyze customer reviews and identify positive and negative sentiments, so please convert the text reviews into vectors."),
    get_detailed_instruct(task, "Instruction: I am working on a project where I need to cluster similar images of street art. How can I get the relevant feature vectors from these images for clustering?"),
    get_detailed_instruct(task, "Instruction: My dog is always getting into things that can be dangerous. Can you help me identify the object my dog is about to eat?")
]

documents = [
    "Text language model",
    "Text embedding",
    "Image feature vector",
    "Image classification",
    "Image Frame Interpolation",
    "Image pose detection",
    "Audio embedding",
    "Text preprocessing",
    "Image object detection",
    "Video classification",
    "Audio Speech-to-Text",
    "Audio event classification",
    "Text classification",
    "Image segmentation"
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('/data/zhangzk/Workspace/Models/gte-qwen-7B')
model = AutoModel.from_pretrained('/data/zhangzk/Workspace/Models/gte-qwen-7B', trust_remote_code=False)

max_length = 8192

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:4] @ embeddings[4:].T)
top_k_indices = torch.topk(scores, k=3, dim=1).indices
print(top_k_indices)