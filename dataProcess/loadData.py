import json
import random

import numpy as np

import constant


def load(path, level=-1, lower=True):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for row in f:
            ins = json.loads(row)
            labels = ins['mention']['labels']
            baselevel = len(labels)
            baselabel = labels[baselevel - 1]
            if baselevel < level:
                padding = '/padding'
                if not lower:
                    padding = padding.upper()
                for i in range(level - baselevel):
                    padlabel = baselabel + padding * (i + 1)
                    labels.append(padlabel)
            ins['mention']['labels'] = labels
            data.append(ins)
    return data


def getTestValSplit(data, seed, val_path=None,label_path=None,val_ratio=0.2,active_ratio=0.2):
    random.seed(seed)
    typeset = set()
    type_idxes = dict()
    for i, ins in enumerate(data):
        labels = ins['mention']['labels']
        level = len(labels)
        _type = labels[level - 1]
        if _type not in typeset:
            typeset.add(_type)
            type_idxes[_type] = [i]
        else:
            type_idxes[_type].append(i)
    data = np.asarray(data)
    val_idxes = []
    for k, v in type_idxes.items():
        # print(f'{k} : {len(v) * val_ratio}')
        if len(v) * active_ratio < 1:
            continue
        size = round(len(v) * active_ratio)
        # print(f'size: {size}')
        val_idxes.extend(random.sample(v, size))

    val_mask = np.zeros(len(data), dtype=bool)
    val_mask[val_idxes] = True
    val_data = data[val_mask]
    for ins in val_data:
        ins['mention']['dtype'] = "extra"
    lab_data = data[~val_mask]

    with open(val_path, "w", encoding="utf-8") as f_val:

        for val_line in list(val_data):
            f_val.write(json.dumps(val_line, ensure_ascii=False) + "\n")

    with open(label_path, "w", encoding="utf-8") as f_lab:

        for lab_line in list(lab_data):
            f_lab.write(json.dumps(lab_line, ensure_ascii=False) + "\n")
    return list(lab_data), list(val_data)


def loadDataset(dataset, seed, split_val=0, val_ratio=0.2,active_budget=0.2):
    paths = constant.split_dataPath[dataset]
    if dataset == 'BBN':
        level = 2
        lower = False
    elif dataset == 'OntoNotes':
        level = 3
        lower = True
    elif dataset == 'FewNerd':
        level = 2
        lower = True
    else:
        raise Exception(f'not support dataset {dataset}')
    # labeled_data = load(paths[0], level, lower)
    labeled_known = load(paths[0], level, lower)
    val_data = load(paths[1], level, lower)
    unlabeled_known_data = load(paths[2], level, lower)
    unknown_data = load(paths[3], level, lower)
    extra_known = load(paths[4], level, lower)

    # extra_path = 'dataset/stdData/'+dataset+'/split/extra_known.json'
    # label_path = 'dataset/stdData/'+dataset+'/split/labeled.json'
    # labeled_known, extra_known = getTestValSplit(labeled_data, seed, extra_path, label_path, active_ratio=active_budget)



    print('labled_known_data')
    print(len(labeled_known))
    print('extra_known_data')
    print(len(extra_known))
    print('unlabeled_known_data')
    print(len(unlabeled_known_data))
    print('val_data')
    print(len(val_data))
    print('unknown_data')
    print(len(unknown_data))
    unlabeled_data = unknown_data + unlabeled_known_data

    # val_path = 'dataset/stdData/'+dataset+'/split/val.json'
    # label_path = 'dataset/stdData/'+dataset+'/split/label.json'
    # labeled_known, val_data = getTestValSplit(labeled_data, seed, val_path,label_path,val_ratio=val_ratio)

    # if split_val == 1:
    return unlabeled_data, labeled_known, val_data,extra_known
    # else:
    #     return unlabeled_data, labeled_data, None
