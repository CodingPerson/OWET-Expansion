import json
from collections import Counter
from scipy.optimize import linear_sum_assignment as linear_assignment

import numpy as np
import tqdm

from openai import OpenAI

data_dir = 'dataset/stdData'
data_level = {'BBN':2, 'OntoNotes':3}
known_num = {'BBN':47, 'OntoNotes':74}
unknown_num = {'BBN':14, 'OntoNotes':20}
known_types = []
unknown_types = []

dataset = 'OntoNotes'
batch_size = 1
api = 'openai/gpt'
post_name = '4.1-nano'  # 4.1 4.1-mini 4.1-nano 4o-mini

type_path = f'{data_dir}/{dataset}/split/type2id_pad.txt'
with open(type_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    all_types = []
    for line in lines:
        line = line.strip().split(' ')
        type_name = line[0].lower()
        all_types.append(type_name)

tmp = all_types[:known_num[dataset]]
for t in tmp:
    if t.count('/') != data_level[dataset]:
        continue
    t = t.replace('/padding', '')
    known_types.append(t)
tmp = all_types[known_num[dataset]:]
for t in tmp:
    if t.count('/') != data_level[dataset]:
        continue
    t = t.replace('/padding', '')
    unknown_types.append(t)

type2id = {}
for i, t in enumerate(known_types):
    type2id[t] = i
for i, t in enumerate(unknown_types):
    type2id[t] = i + len(known_types)

known_types_info = '{'
for i, t in enumerate(known_types):
    if i == 0:
        known_types_info += t
    else:
        known_types_info += f',{t}'
known_types_info += '}'

data = []
labels = []
data_nums = []
known_idx = []
id = 0
data_paths = [f'{data_dir}/{dataset}/split/unknown.json', f'{data_dir}/{dataset}/split/unlabeled.json',f'{data_dir}/{dataset}/split/labeled.json',f'{data_dir}/{dataset}/split/extra_known.json']
for data_path in data_paths:
    with open(data_path, 'r', encoding='utf-8') as f:
        for row in f:
            ins = json.loads(row)
            sentence = ins['sentence']
            start = ins['mention']['start']
            end = ins['mention']['end']
            mention = sentence[start:end]
            label = ins['mention']['labels'][-1].lower()
            label = label.replace('/padding', '')

            sentence = ' '.join(sentence)
            mention = ' '.join(mention)
            instance = f'Sentence: {sentence}\nMention: {mention}'

            data.append(instance)
            labels.append(type2id[label])
            if 'labeled' in data_path:
                known_idx.append(id)
            id+=1
            # labels.append(label)
    data_nums.append(len(data))

# generate gpt prediction
if api == 'openai/gpt':
    key = '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'
    model = OpenAI(api_key=key,base_url="https://openrouter.ai/api/v1")
    model_name = f'openai/gpt-{post_name}'
elif api == 'deepseek':
    key = 'sk-ad804aeab035488aa2a0d9f5ed38ef39'
    model = OpenAI(api_key=key, base_url='https://api.deepseek.com')
    model_name = 'deepseek-reasoner'

print(model_name)
print(f'tot data num: {len(data)}')
outputs = []
count = 0
start = 0
for i in tqdm.tqdm(range(start, len(data), 1),total=(len(data)-start)):
    batch_message = data[i]
    if i in known_idx:
        outputs.append(labels[i])
        continue
    messages = [
        {
            "role": "system",
            "content":
                f"You are an AI assistant who specializes entity types. Your task is as follows: according to the sentence, "
                f"predict the entity type of entity mention in the sentence. If the predicted type belongs to the known types "
                f"supported by the system, return the corresponding known type; otherwise, suggest the most likely unknown "
                f"type name(couldn't be 'unknown type', you should give a actual type name different from known types). "
                f"The supported known types include: {known_types_info}.\n"
                "Only provide one type from the above known "
                f"types, or suggest an unknown type name, and do not give the explanation. \n"
                f"The input format is as follows::\n"
                f"\nSentence: <sentence>\nMention: <mention>\n"
        },
        {
            "role": "user",
            "content": batch_message
        }
    ]
    response = model.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        timeout=180,
        stream=True,
    )
    info = response.choices[0].message.content
    info = json.loads(info)
    if not info['type'].startswith('/'):
        info['type'] = '/' + info['type']
    outputs.append(info['type'])
    if len(outputs) == 100:
        count += 1
        print(count)
        print(i)
        with open(f'{dataset}_{model_name}_predict.txt', 'a', encoding='utf-8') as f:
            for output in outputs:
                f.write(output)
                f.write('\n')
        outputs = []

with open(f'{dataset}_{model_name}_predict.txt', 'a', encoding='utf-8') as f:
    for output in outputs:
        f.write(output)
        f.write('\n')