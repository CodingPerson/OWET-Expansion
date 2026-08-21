import json
import random

import torch
import tqdm
from sentence_transformers import SentenceTransformer

# 相似度top5
model_path = '.modelfile/sentenceBert/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130'
model = SentenceTransformer(model_path)
model.to('cuda:0')

data_dir = 'dataset/stdData'
dataset = 'OntoNotes'
batch_size = 16

labeled_data = []
unlabeled_data = []

with open(f'{data_dir}/{dataset}/split/labeled.json', 'r', encoding='utf-8') as f:
    for row in f:
        ins = json.loads(row)
        sentence = ins['sentence']
        start = ins['mention']['start']
        end = ins['mention']['end']
        mention = sentence[start:end]

        sentence = ' '.join(sentence)
        mention = ' '.join(mention)
        instance = f'{mention} , {sentence}'
        labeled_data.append(instance)

data_paths = [f'{data_dir}/{dataset}/split/unlabeled.json', f'{data_dir}/{dataset}/split/unknown.json',f'{data_dir}/{dataset}/split/labeled.json',f'{data_dir}/{dataset}/split/extra_known.json']
for data_path in data_paths:
    with open(data_path, 'r', encoding='utf-8') as f:
        for row in f:
            ins = json.loads(row)
            sentence = ins['sentence']
            start = ins['mention']['start']
            end = ins['mention']['end']
            mention = sentence[start:end]

            sentence = ' '.join(sentence)
            mention = ' '.join(mention)
            instance = f'{mention} , {sentence}'
            unlabeled_data.append(instance)
print('load data done!')

def get_cls_embedding(texts, model, tokenizer, batch_size=64):
    cls_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(texts), batch_size), total=len(texts) // batch_size + 1):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            inputs = {key:val.to('cuda:0') for key, val in inputs.items()}
            outputs = model(inputs)['token_embeddings']
            cls_embeddings.append(outputs[:,0,:])
        cls_embeddings = torch.cat(cls_embeddings,dim=0)
    return cls_embeddings

labeled_cls_embeddings = get_cls_embedding(labeled_data, model, model.tokenizer, batch_size)
unlabeled_cls_embeddings = get_cls_embedding(unlabeled_data, model, model.tokenizer, batch_size)

print(len(labeled_data))
print(len(unlabeled_data))
print(len(labeled_cls_embeddings))
print(len(unlabeled_cls_embeddings))
print('encode done!')

similarity = model.similarity(unlabeled_cls_embeddings, labeled_cls_embeddings)
print('similarity done!')

topk_values, topk_indices = torch.topk(similarity, k=5, dim=1)
print('topk done!')

topk_indices = topk_indices.cpu().numpy().tolist()

with open(f'{dataset}_top5.json', 'w', encoding='utf-8') as f:
    tmp_data = []
    for i, idxes in enumerate(topk_indices):
        tmp_data.append(idxes)
    json.dump(tmp_data, f)
print('done!')

# # 随机top5
# dataset = 'OntoNotes'
# tot_num = {'BBN':5795, 'OntoNotes': 4368}
# data_num = {'BBN':7487, 'OntoNotes': 6106}
#
# # with open(f'{dataset}_top5_random.json', 'r', encoding='utf-8') as f:
# #     topk_indices = json.load(f)
#
# topk_indices = []
# for _ in range(data_num[dataset]):
#     idxes = random.sample(range(tot_num[dataset]), 5)
#     topk_indices.append(idxes)
#
# with open(f'{dataset}_top5_random.json', 'w', encoding='utf-8') as f:
#     json.dump(topk_indices, f)