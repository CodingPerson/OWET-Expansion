import argparse
import random
from collections import Counter

import numpy as np
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class OWETDataset(Dataset):
    def __init__(self, data, type2id, tokenizer, num_classes, args: argparse.Namespace, mode='labeled'):
        self.mode = mode
        self.data = data
        self.type2id = type2id
        self.tokenizer = tokenizer

        self.len_max_tokenization = 0
        self.inputs = []
        self.final_inputs = []
        self.args = args

        self.targets = np.zeros([len(self.data), num_classes], np.float32)
        self.top_types = []  # 每个样本的top-level type
        self.sample_weights = []  # 采样权重
        self.preprocess()

    def preprocess(self):
        # {'sentence':[],'mention':{'start':idx,'end':idx,'labels':[],'dtype':xxx}}
        for i, ins in tqdm.tqdm(enumerate(self.data), desc='preprocess data', total=len(self.data)):
            context = ins['sentence']
            if self.args.do_lower:
                context = [word.lower() for word in context]
            start = ins['mention']['start']
            end = ins['mention']['end']
            mention = context[start:end]
            leftctx = context[:start]
            rightctx = context[end:]

            if self.mode == 'labeled':
                label_id = [self.type2id[label] for label in ins['mention']['labels'] if label in self.type2id]
                self.top_types.append(label_id[-1])
                for id_ in label_id:
                    self.targets[i, id_] = 1.0

            if len(mention) > self.args.mention_limit:
                mention = mention[:self.args.mention_limit]
            mention_ = ' '.join(mention)
            context_ = ' '.join(leftctx + mention + rightctx)
            len_tokenization = len(self.tokenizer.encode_plus(mention_, context_)["input_ids"])
            if len_tokenization > self.args.input_limit:
                over = len_tokenization - self.args.input_limit
                rightctx = rightctx[:-over]
            context_ = ' '.join(leftctx + mention + rightctx)
            len_tokenization = len(self.tokenizer.encode_plus(mention_, context_)["input_ids"])
            self.len_max_tokenization = max(self.len_max_tokenization, len_tokenization)
            self.inputs.append([mention_, context_])

        type2num = Counter(self.top_types)
        self.sample_weights = [1 / type2num[t] for t in self.top_types]
        self.sample_weights = torch.DoubleTensor(self.sample_weights)

        self.targets = torch.from_numpy(self.targets)
        for ins in tqdm.tqdm(self.inputs, desc='encode data for labeled', total=len(self.inputs)):
            inputs = self.tokenizer.encode_plus(
                ins[0], ins[1],
                add_special_tokens=True,
                # max_length=min(self.args.input_limit, self.len_max_tokenization),
                # truncation_strategy="only_second",
                # padding=min(self.args.input_limit, self.len_max_tokenization),
                return_tensors="pt",
                # truncation=True
            )
            self.final_inputs.append(inputs)

    def __getitem__(self, index):
        # print(self.inputs[index])
        # print(index)
        # print(self.data[index]['mention']['labels'])
        return self.final_inputs[index], self.targets[index], index

    def __len__(self):
        return len(self.data)
class ActiveDataset(Dataset):
    def __init__(self, data, type2id, tokenizer, num_classes, args: argparse.Namespace, mode='active'):
        self.mode = mode
        self.data = data
        self.type2id = type2id
        self.tokenizer = tokenizer

        self.len_max_tokenization = 0
        self.inputs = []
        self.final_inputs = []
        self.args = args

        self.targets = np.zeros([len(self.data), num_classes], np.float32)
        self.top_types = []  # 每个样本的top-level type
        self.sample_weights = []  # 采样权重
        self.preprocess()

    def preprocess(self):
        # {'sentence':[],'mention':{'start':idx,'end':idx,'labels':[],'dtype':xxx}}
        for i, ins in tqdm.tqdm(enumerate(self.data), desc='preprocess data', total=len(self.data)):
            context = ins['sentence']
            if self.args.do_lower:
                context = [word.lower() for word in context]
            start = ins['mention']['start']
            end = ins['mention']['end']
            mention = context[start:end]
            leftctx = context[:start]
            rightctx = context[end:]

            label_id = self.type2id[ins['mention']['labels'][-1]]
            self.top_types.append(label_id)
            self.targets[i, label_id] = 1.0

            if len(mention) > self.args.mention_limit:
                mention = mention[:self.args.mention_limit]
            mention_ = ' '.join(mention)
            context_ = ' '.join(leftctx + mention + rightctx)
            len_tokenization = len(self.tokenizer.encode_plus(mention_, context_)["input_ids"])
            if len_tokenization > self.args.input_limit:
                over = len_tokenization - self.args.input_limit
                rightctx = rightctx[:-over]
            context_ = ' '.join(leftctx + mention + rightctx)
            len_tokenization = len(self.tokenizer.encode_plus(mention_, context_)["input_ids"])
            self.len_max_tokenization = max(self.len_max_tokenization, len_tokenization)
            self.inputs.append([mention_, context_])

        type2num = Counter(self.top_types)
        self.sample_weights = [1 / type2num[t] for t in self.top_types]
        self.sample_weights = torch.DoubleTensor(self.sample_weights)

        self.targets = torch.from_numpy(self.targets)
        for ins in tqdm.tqdm(self.inputs, desc='encode data for labeled', total=len(self.inputs)):
            inputs = self.tokenizer.encode_plus(
                ins[0], ins[1],
                add_special_tokens=True,
                # max_length=min(self.args.input_limit, self.len_max_tokenization),
                # truncation_strategy="only_second",
                # padding=min(self.args.input_limit, self.len_max_tokenization),
                return_tensors="pt",
                # truncation=True
            )
            self.final_inputs.append(inputs)

    def __getitem__(self, index):
        # print(self.inputs[index])
        # print(index)
        # print(self.data[index]['mention']['labels'])
        return self.final_inputs[index], self.targets[index], index

    def __len__(self):
        return len(self.data)

def labeled_collate_fn(batch):
    inputs, targets, indexes = zip(*batch)

    input_ids = [item['input_ids'].squeeze(0) for item in inputs]
    attention_mask = [item['attention_mask'].squeeze(0) for item in inputs]
    token_type_ids = [item['token_type_ids'].squeeze(0) for item in inputs]

    # 填充到统一长度
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

    # 将 targets 转换为张量
    targets = torch.stack(targets)
    indexes = torch.tensor(indexes, dtype=torch.int)

    return input_ids, attention_mask, token_type_ids, targets, indexes

