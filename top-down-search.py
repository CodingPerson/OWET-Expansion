import copy
import csv
import os
import parser
from collections import Counter, defaultdict
from itertools import combinations

from sklearn.cluster import KMeans

from config import initConfig
from train import getClusterResult, getclusterInfo, logClusterMatch, getInstanceCounts, getPathMatch, maxMatchScore, \
getConceptCounts

os.environ['PYTHONHASHSEED'] = '11'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

torch.use_deterministic_algorithms(True)
import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from sympy.physics.control.control_plots import np
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

import constant
from dataProcess.OWETDataset import OWETDataset, labeled_collate_fn
from dataProcess.loadData import loadDataset
from pretrain import toEval
from models.model import OWETModel, Cluster2Box, cluster2box
from utils.cluster_acc import cluster_acc, log_accs_from_preds, hdbscan_acc, test_agglo, finch_acc
from utils.finch import FINCH
from utils.hdbscanTools import hdbscanManager, MSTLinked, compute_mutual_reachability
from utils.metric import getVmAndARIAndNMI, getB3Eval
from utils.utils import loadCheckConfig, getType2Id, getLevelTarget, setSeed, printConfig, getTypesInput, getTypeNum, \
    getInstanceCounts_unknown, getTypesDesc
from utils.view_generator import view_generator
from search_util import ClusterMerge, ClusterFilter


# def getClusterResult(model, dataloader, device, id2type, args):
#     model.eval()
#     with torch.no_grad():
#         all_feats = []
#         all_idxes = []
#         targets = []
#         mask_lab = np.array([])
#         mask_unlab_known = np.array([])
#         bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
#                         desc='extra feature')
#         for step, batch in bar:
#             batch = tuple(t.to(device) for t in batch)
#             input_ids, input_mask, segment_ids, label_ids, idxes = batch
#             feats, _ = model(input_ids, segment_ids, input_mask, label_ids, mode="train")
#
#             all_feats.extend(feats.cpu().numpy())
#             all_idxes.extend(idxes.cpu().numpy().tolist())
#             targets.extend(label_ids.cpu().numpy())
#             del feats, input_mask, segment_ids
#             mask_lab = np.append(mask_lab, np.array(
#                 [True if dataloader.dataset.data[i]['mention']['dtype'] == 'known_labeled' else False
#                  for i in idxes]))
#             mask_unlab_known = np.append(mask_unlab_known, np.array(
#                 [True if dataloader.dataset.data[i]['mention']['dtype'] == 'known_unlabeled' else False
#                  for i in idxes]))
#
#         targets = np.asarray(targets)
#         all_feats = np.asarray(all_feats)
#         level_target = getLevelTarget(targets, args.data_level, id2type)
#         mask_lab = mask_lab.astype(bool)
#         mask_unlab_known = mask_unlab_known.astype(bool)
#
#         # cluster acc
#         if args.cluster_type == 'HAC':
#             linked = linkage(all_feats, method=args.cluster_method)
#         elif args.cluster_type == 'HDBSCAN':
#             dist_martix = compute_mutual_reachability(all_feats, min_samples=5)
#             linked = linkage(dist_martix, method=args.cluster_method)
#         else:
#             raise Exception(f'args.cluster_type error {args.cluster_type}')
#
#         clusterAccDict = dict()
#         print('getting cluster result...')
#         level_acc, best_level_k, level_preds = test_agglo(linked, level_target, mask_lab, mask_unlab_known, args,
#                                                           rePred=True, onlyEnd=False)
#         print(level_acc)
#         print(best_level_k)
#         preds = level_preds[-1]
#         lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc = level_acc[args.data_level - 1]
#         best_k = best_level_k[args.data_level - 1]
#         clusterAccDict[0] = [tot_acc, unlab_known_acc, unlab_unknown_acc, lab_acc, best_k, best_level_k]
#
#         target = level_target[args.data_level - 1]
#         # labeled
#         target_lab = target[mask_lab]
#         pred_lab = preds[mask_lab]
#         # unlabeled
#         target_unlab = target[~mask_lab]
#         pred_unlab = preds[~mask_lab]
#         mask_unlab_known = mask_unlab_known[~mask_lab]
#
#         targetList = [target_lab, target_unlab]
#         predList = [pred_lab, pred_unlab]
#         target_unlab_known = target_unlab[mask_unlab_known]
#         pred_unlab_known = pred_unlab[mask_unlab_known]
#         target_unlab_unknown = target_unlab[~mask_unlab_known]
#         pred_unlab_unknown = pred_unlab[~mask_unlab_known]
#         targetList.extend([target_unlab_known, target_unlab_unknown])
#         predList.extend(([pred_unlab_known, pred_unlab_unknown]))
#
#         # B3
#         b3Dict = getB3Eval(targetList, predList)
#         # V_measure and ARI
#         VmARINMIDict = getVmAndARIAndNMI(targetList, predList)
#
#         clusterResult = dict()
#         print('getting cluster info...')
#         ins2cluster, centroids, ins2pbl,cluster_ins_re_dict = getclusterInfo(all_feats, preds, args.pbl_ratio, args.enable_pbl_ratio)
#         clusterResult['ins2cluster'] = ins2cluster
#         clusterResult['centroids'] = centroids
#         clusterResult['ins2pbl'] = torch.from_numpy(ins2pbl).bool()
#         clusterResult['cluster_ins_re_dict']=cluster_ins_re_dict
#         return clusterResult, all_idxes, all_feats, level_preds, (clusterAccDict, b3Dict, VmARINMIDict), mask_lab


myargs = initConfig()
path = myargs.TR_path
data = myargs.dataset

args = loadCheckConfig(path+'/config.pkl')
# args = loadCheckConfig('log/train_BBN_20260513_172238/config.pkl')
args.device = 0
if data=='OntoNotes':
    checkpoint = torch.load(
        path+'/checkpoint_OntoNotes_lr_2e-05_latest.pth',
        weights_only=True,map_location=torch.device('cuda:'+str(args.device)))
elif data=='BBN':
    checkpoint = torch.load(
        path + '/checkpoint_BBN_lr_2e-05_latest.pth',
        weights_only=True, map_location=torch.device('cuda:' + str(args.device)))

save_dir = path
for key, value in vars(args).items():
    print(f"{key}:{value}")
setSeed(args.seed, args.n_gpu)

args.single_box_model = 0
args.cpu_g = torch.Generator(device='cpu').manual_seed(args.seed)
args.gpu_g = torch.Generator(device=f'cuda:{args.device}').manual_seed(args.seed)
args.enable_pbl_ratio = 0
args.pbl_ratio = 0

type2id = getType2Id(constant.type2id_path[args.dataset][1])
id2type = {v: k for k, v in type2id.items()}

model = OWETModel(args)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

#---------------
types, types_input = getTypesInput(constant.type2id_path[args.dataset][1], args.num_known_class, model.tokenizer,
                                       None)
unknown_set = constant.unknownSet[args.dataset]
type_desc = getTypesDesc(constant.type_description[args.dataset])
types_all = []
with open(constant.type2id_path[args.dataset][1], 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split()
        line = line[0]
        types_all.append(line)
#---------------

# 加载数据
unlabeled_data, labeled_data, val_data, extra_known = loadDataset(args.dataset, args.seed, args.split_val,
                                                                  args.val_ratio, args.active_budget)
args.total_budget = len(extra_known)
labeled_dataset = OWETDataset(labeled_data, type2id, model.tokenizer, num_classes=args.num_known_class, args=args,
                              mode='labeled')
labeled_g = torch.Generator().manual_seed(args.seed)
args.labeled_g = labeled_g
sampler = torch.utils.data.WeightedRandomSampler(labeled_dataset.sample_weights, num_samples=len(labeled_dataset),
                                                 generator=labeled_g)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.labeled_batch_size, shuffle=False, sampler=sampler,
                                drop_last=True, collate_fn=labeled_collate_fn, num_workers=0, generator=labeled_g)
if args.split_val == 1:
    args.num_val_class = getTypeNum(val_data)
    val_dataset = OWETDataset(val_data, type2id, model.tokenizer, num_classes=args.num_known_class, args=args,
                              mode='labeled')
    sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_g = torch.Generator().manual_seed(args.seed)
    args.val_g = val_g
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                drop_last=False, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                generator=val_g)
else:
    val_dataloader = None

adapted_data = unlabeled_data + labeled_data + extra_known
adapted_dataset = OWETDataset(adapted_data, type2id, model.tokenizer, num_classes=args.num_known_class,
                              args=args,
                              mode='unlabeled')
adapted_g = torch.Generator().manual_seed(args.seed)
args.adapted_g = adapted_g
sampler = torch.utils.data.RandomSampler(adapted_dataset, num_samples=len(adapted_dataset), generator=adapted_g)
adapted_dataloader = DataLoader(adapted_dataset, batch_size=args.mlm_batch_size, shuffle=False,
                                drop_last=True, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                generator=adapted_g)

# eval的dataloader
total_data = unlabeled_data + labeled_data + extra_known
eval_dataset = OWETDataset(total_data, type2id, model.tokenizer,
                           num_classes=args.num_known_class + args.num_unknown_class, args=args,
                           mode='labeled')
sampler = torch.utils.data.SequentialSampler(eval_dataset)
eval_g = torch.Generator().manual_seed(args.seed)
args.eval_g = eval_g
eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             drop_last=False, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                             generator=eval_g)

# clusterResult, all_idxes, all_feats, level_preds, cluster_acc, mask_lab = getClusterResult_test(model, eval_dataloader,
#                                                                   device, id2type, val_dataloader,args)

clusterResult, all_idxes, all_feats, level_preds, re, mask_lab=getClusterResult(model, eval_dataloader,device, id2type,args,total_data)

predicts = level_preds[-1]
result = clusterResult
torch_feats = torch.from_numpy(all_feats).to(device)

torch_feats = torch.nn.functional.normalize(torch_feats, dim=-1)
epoch_feats = torch_feats

#---------------
type_counts = getConceptCounts(types, type2id, labeled_dataloader.dataset.targets, level=args.data_level)
ins_counts, type_count_all = getInstanceCounts(eval_dataset.targets, all_idxes, level_preds, mask_lab, types,
                                                       type2id, id2type)
mask_unlabeled = ~mask_lab
type_counts_unlabeled = getConceptCounts(types_all, type2id, eval_dataloader.dataset.targets[mask_unlabeled], level=args.data_level)
ins_counts_unlabeled = getInstanceCounts_unknown(eval_dataset.targets, all_idxes, level_preds, mask_unlabeled, types_all,
                                                       type2id, id2type)
#---------------

matrix_unlabeled, _ = getPathMatch(type_counts_unlabeled, ins_counts_unlabeled)
match_result_unlabeled, _ = maxMatchScore(matrix_unlabeled)
matrix, _ = getPathMatch(type_counts, ins_counts)
match_result, _ = maxMatchScore(matrix)

concept_type_path_all, ins_type_path = type_counts_unlabeled['type_path'], ins_counts_unlabeled['type_path']


unknown_cluster_filter,unknown_id_label = ClusterFilter(clusterResult,all_idxes,all_feats,match_result,match_result_unlabeled,concept_type_path_all,ins_type_path,id2type,types,types_all,type_count_all)
print('filter\n')
print(unknown_id_label)
if 'OntoNotes' in save_dir:
    child_types=[]
    for type,desc in type_desc.items():
        if '/other/' in type and type not in unknown_set:
            child_types.append(type)
    child_types=' '.join(child_types)
    type_desc['/other'] = type_desc['/other']+f"\nThese entities may be loosely associated with more specific subareas such as {child_types}"

known_type_desc=defaultdict()
for type,desc in type_desc.items():
    if type not in unknown_set:
        known_type_desc[type]=desc
unknown_predict = ClusterMerge(
    clusterResult,
    all_idxes,
    all_feats,
    match_result,
    match_result_unlabeled,
    concept_type_path_all,
    ins_type_path,
    id2type,
    type2id,
    types,
    unknown_cluster_filter,
    unknown_id_label,
    known_type_desc
)
# print(f'inserted unknown clusters: {len([p for p in unknown_predict if len(p) > 0])}')
# logClusterMatch(match_result,concept_type_path_all,ins_type_path,type_count_all,id2type,f'{save_dir}/ClusterMatch.txt', 0)







