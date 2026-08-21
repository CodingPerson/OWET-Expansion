import copy
import csv
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
import math
from sklearn.cluster import KMeans

from train import getClusterResult, getInBoxRate, getclusterInfo, logClusterMatch, getInstanceCounts, getPathMatch, maxMatchScore, \
getConceptCounts

os.environ['PYTHONHASHSEED'] = '11'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
import torch
from openai import OpenAI
torch.use_deterministic_algorithms(True)
import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from sympy.physics.control.control_plots import np
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM
import string
import constant
from dataProcess.OWETDataset import OWETDataset, labeled_collate_fn
from dataProcess.loadData import loadDataset
from pretrain import toEval
from models.model import OWETModel, Cluster2Box, cluster2box
from utils.cluster_acc import cluster_acc, log_accs_from_preds, hdbscan_acc, test_agglo, finch_acc
from utils.finch import FINCH
from utils.hdbscanTools import hdbscanManager, MSTLinked, compute_mutual_reachability
from utils.metric import getVmAndARIAndNMI, getB3Eval
from utils.utils import loadCheckConfig, getType2Id, getLevelTarget, setSeed, printConfig, getTypesInput, getTypeNum
from utils.view_generator import view_generator

import copy
from collections import defaultdict


def build_cluster_to_concept_map(match_result, concept_type_path, ins_type_path):
    """
    构建：cluster level-1 → concept type path
    """
    cluster_map = {}
    for c_idx, i_idx in match_result:
        cluster_id = ins_type_path[i_idx][0]
        cluster_map[cluster_id] = concept_type_path[c_idx]
    return cluster_map


def refine_instance_type_path(ins_type_path, cluster_map, level_num):
    """
    根据 known match 结果，refine instance type path
    unknown cluster → 后续层设为 -1
    """
    refined = copy.deepcopy(ins_type_path)

    for path in refined:
        cluster_id = path[0]

        if cluster_id not in cluster_map:
            path[1:] = [-1] * (level_num - 1)
        else:
            path[1:] = cluster_map[cluster_id][1:]

    return refined


def get_unknown_clusters(refined_paths):
    """
    找到 unknown clusters（第二层为 -1）
    """
    unknown_clusters = set()
    for path in refined_paths:
        if path[1] == -1:
            unknown_clusters.add(path[0])
    return sorted(list(unknown_clusters))


def compute_unknown_type_distribution(cluster_id, type_count_all, id2type, known_types, level_num):
    """
    统计一个 cluster 内 unknown type（含父节点）分布
    """
    cluster_counts = type_count_all[level_num - 1][cluster_id]

    total = sum(cluster_counts.values())
    type_counter = defaultdict(int)

    for type_id, count in cluster_counts.items():
        type_name = id2type[type_id]

        if type_name not in known_types:
            # 当前 type
            type_counter[type_name] += count

            # 父节点累计
            for l in range(1, level_num):
                parent = type_name.rsplit('/', l)[0]
                if parent not in known_types:
                    type_counter[parent] += count

    return type_counter, total


def select_representative_clusters(
    unknown_clusters,
    type_count_all,
    id2type,
    known_types,
    all_types,
    level_num,
    threshold=0.5
):
    """
    核心筛选逻辑：
    每个 unknown type 选一个最优 cluster
    """
    unknown_types = list(set(all_types) - set(known_types))

    best_cluster_per_type = {
        t: {"cluster": None, "rate": 0.0, "count": 0}
        for t in unknown_types
    }

    selected_clusters = set()

    for cluster_id in unknown_clusters:

        type_dist, total = compute_unknown_type_distribution(
            cluster_id, type_count_all, id2type, known_types, level_num
        )

        if total == 0:
            continue

        for t in unknown_types:
            count = type_dist.get(t, 0)
            rate = count / total

            if rate < threshold:
                continue

            best = best_cluster_per_type[t]

            # 选择更优 cluster（rate优先，其次count）
            if (best["cluster"] is None or
                rate > best["rate"] or
                (rate == best["rate"] and count > best["count"])):

                if best["cluster"] is not None:
                    selected_clusters.discard(best["cluster"])

                best_cluster_per_type[t] = {
                    "cluster": cluster_id,
                    "rate": rate,
                    "count": count
                }

                selected_clusters.add(cluster_id)

    return selected_clusters, best_cluster_per_type


def build_unknown_label_map(best_cluster_per_type,level_num):
    """
    构建 cluster → unknown type label
    """
    cluster_label_map = defaultdict(list)

    for t, info in best_cluster_per_type.items():
        if info["cluster"] is not None:
            cluster_label_map[info["cluster"]].append(t)

    # 加 root + 清理 padding
    for cluster_id, labels in cluster_label_map.items():
        # labels = ['root'] + labels
        re_labels=set()
        for l in labels:
            # if 'padding' in l:
            #     re_labels.add(l.replace("/padding",""))
            # else:
                re_labels.add(l)
        if len(re_labels) > 1:
            re_labels=[l for l in re_labels if l.count("/")==level_num]
        cluster_label_map[cluster_id] = list(re_labels)

    return cluster_label_map


def ClusterFilter(
    clusterResult,
    all_idxes,
    all_feats,
    match_result,
    match_result_unlabeled,   # 目前没用，可以后续扩展
    concept_type_path,
    ins_type_path,
    id2type,
    types,
    types_all,
    type_count_all
):
    """
    主函数（重构后）
    """

    level_num = len(concept_type_path[0])

    # 1️⃣ 构建 known cluster → concept mapping
    cluster_map = build_cluster_to_concept_map(
        match_result, concept_type_path, ins_type_path
    )

    # 2️⃣ refine instance type path
    refined_paths = refine_instance_type_path(
        ins_type_path, cluster_map, level_num
    )

    # 3️⃣ 找 unknown clusters
    unknown_clusters = get_unknown_clusters(refined_paths)

    # 4️⃣ 筛选代表 cluster
    selected_clusters, best_cluster_per_type = select_representative_clusters(
        unknown_clusters,
        type_count_all,
        id2type,
        types,
        types_all,
        level_num
    )

    # 5️⃣ 构建 label 映射
    unknown_id_label = build_unknown_label_map(best_cluster_per_type,level_num)

    return selected_clusters, unknown_id_label

# def ClusterFilter(clusterResult, all_idxes, all_feats, match_result,match_result_unlabeled,concept_type_path,ins_type_path,id2type,types,types_all,type_count_all):
#     cluster_ins_re_dict = clusterResult['cluster_ins_re_dict']
#
#     ## 将known cluster的type path填充好，然后将所有unknown cluster的type path设为-1
#
#     match_dict=defaultdict()
#     for pair in match_result:
#         c_idx = pair[0]
#         i_idx=pair[1]
#         level_1_concept_idx=concept_type_path[c_idx][0]
#         level_1_inscluster_idx = ins_type_path[i_idx][0]
#
#         match_dict[level_1_inscluster_idx]=concept_type_path[c_idx]
#
#     match_dict_gold=defaultdict()
#     for pair in match_result_unlabeled:
#         c_idx = pair[0]
#         i_idx=pair[1]
#         level_1_concept_idx=concept_type_path[c_idx][0]
#         level_1_inscluster_idx = ins_type_path[i_idx][0]
#
#         match_dict_gold[level_1_inscluster_idx]=concept_type_path[c_idx]
#
#     ## 我们在这个地方对concept path进行check
#     concept_path_name=[[] for i in range(len(concept_type_path))]
#     for i in range(len(concept_type_path)):
#         for v in concept_type_path[i]:
#             concept_path_name[i].append(id2type[v])
#
#
#     # for k in match_dict.keys():
#     #     if k not in match_dict_gold.keys():
#     #         print(k)
#
#     # unknown_cluster_idxs=[]
#     # known_cluster_idxs=[]
#     level_num = len(concept_type_path[0])
#     #
#     # for i_p in ins_type_path:
#     #     level_1_inscluster_idx = i_p[0]
#     #     if level_1_inscluster_idx not in match_dict.keys():
#     #         unknown_cluster_idxs.append(level_1_inscluster_idx)
#     #     else:
#     #         known_match_concept_path = match_dict[level_1_inscluster_idx]
#     #         known_cluster_idxs.append(known_match_concept_path)
#     refine_ins_type_path=copy.deepcopy(ins_type_path)
#     for j in range(len(refine_ins_type_path)):
#         re_ins_path = refine_ins_type_path[j]
#         level_1_inscluster_idx = re_ins_path[0]
#         if level_1_inscluster_idx not in match_dict.keys():
#             refine_ins_type_path[j][1:] = [-1]*(level_num-1)
#         else:
#             refine_ins_type_path[j][1:] =  copy.deepcopy(match_dict[level_1_inscluster_idx][1:])
#
#     unknown_cluster_idxs = []
#     unknwn_cluster_rep = []
#     for i_p in refine_ins_type_path:
#         cur_level_idx = i_p[1]
#         pre_level_idx = i_p[0]
#         if cur_level_idx == -1:
#             unknwn_cluster_rep.append(cluster_ins_re_dict[pre_level_idx])
#             unknown_cluster_idxs.append(pre_level_idx)
#
#     final_refine_ins_type_all_count = {i: {} for i in range(level_num)}
#     unknown_cluster_type_count = {i: {} for i in unknown_cluster_idxs}
#
#     unknown_cluster_filter = set()
#     unknown_type_list = list(set(types) ^ set(types_all))
#     already_unknown_type = {t:[] for t in unknown_type_list}
#     for k,v in unknown_cluster_type_count.items():
#         unknown_cluster_type_count[k] = copy.deepcopy(type_count_all[level_num-1][k])
#         unknown_num=0
#         for i,n in unknown_cluster_type_count[k].items():
#             unknown_num += n
#
#
#         ## 这地方需要对每一个unknown type及其父节点的num进行统计
#
#         unknown_types_count_dict = {t:0 for t in unknown_type_list}
#         for i,n in unknown_cluster_type_count[k].items():
#             cur_type = id2type[i]
#
#             if cur_type not in types:
#                 unknown_types_count_dict[cur_type] += n
#                 for l in range(1,level_num):
#                     parent_type = cur_type.rsplit('/', l)[0]
#                     if parent_type not in types:
#                         unknown_types_count_dict[parent_type] += n
#         for t,t_n in unknown_types_count_dict.items():
#
#             rate = float(t_n/unknown_num)
#             if (rate >= 0.5):
#
#
#                 if already_unknown_type[t] == []:
#                     already_unknown_type[t].append(k)
#                     already_unknown_type[t].append(rate)
#                     already_unknown_type[t].append(t_n)
#                     unknown_cluster_filter.add(k)
#                     print(k)
#                     print(t)
#                     print(rate)
#                 else:
#                     if already_unknown_type[t][1] < rate or already_unknown_type[t][2]< t_n:
#                         print('remove')
#                         print(already_unknown_type[t][0])
#                         unknown_cluster_filter.remove(already_unknown_type[t][0])
#                         already_unknown_type[t][0]=(k)
#                         already_unknown_type[t][1]=(rate)
#                         already_unknown_type[t][2]=(t_n)
#                         unknown_cluster_filter.add(k)
#                         print('replace\n')
#                         print(k)
#                         print(rate)
#         print('........................')
#     unknown_filter_list = list(sorted(list(unknown_cluster_filter)))
#     unknown_id_label = {t:[] for t in unknown_filter_list}
#     for k,v in already_unknown_type.items():
#         if v != []:
#             type_id = v[0]
#             unknown_id_label[type_id].append(k)
#     for k,v in unknown_id_label.items():
#         unknown_id_label[k] = ['root']+unknown_id_label[k]
#         if len(v) > 1:
#             for vi in v:
#                 if 'PADDING' in vi or 'padding' in vi:
#                     unknown_id_label[k].remove(vi)
#
#
#     # for i in range(level_num):
#     #     if i == 0:
#     #         final_refine_ins_type_all_count[i] = type_count_all[level_num - i - 1]
#     #     else:
#
#     #         for re_ins_path in refine_ins_type_path:
#     #             ins_idx = re_ins_path[i]
#     #             pre_level_idx = re_ins_path[i-1]
#     #             if ins_idx not in final_refine_ins_type_all_count[i].keys():
#     #                 final_refine_ins_type_all_count[i][ins_idx] = copy.deepcopy(
#     #                     final_refine_ins_type_all_count[i - 1][pre_level_idx])
#     #             else:
#     #                 for key,value in final_refine_ins_type_all_count[i-1][pre_level_idx].items():
#     #                     if key not in final_refine_ins_type_all_count[i][ins_idx].keys():
#     #                         final_refine_ins_type_all_count[i][ins_idx][key]=value
#     #                     else:
#     #                         final_refine_ins_type_all_count[i][ins_idx][key]+=value
#
#     return unknown_cluster_filter,unknown_id_label
    
    


def pairwise_cosine_similarity_score(arr1, arr2, keep_percent=1):
# Step 1: normalize arr1
    arr1_norm = arr1 / (np.linalg.norm(arr1, axis=1, keepdims=True) + 1e-8)

    # Step 2: compute center of arr1
    center = np.mean(arr1_norm, axis=0)
    center_norm = center / (np.linalg.norm(center) + 1e-8)

    # Step 3: compute similarity to center
    sims_to_center = np.dot(arr1_norm, center_norm)  # shape: (A,)

    # Step 4: select top-K% closest to center
    A = arr1.shape[0]
    top_k = max(1, int(A * keep_percent))
    top_indices = np.argsort(sims_to_center)[-top_k:]

    # Step 5: filter arr1 by selected indices
    arr1_filtered = arr1[top_indices]

    # Step 6: normalize arr2
    arr2_norm = arr2 / (np.linalg.norm(arr2, axis=1, keepdims=True) + 1e-8)

    # Step 2: compute center of arr1
    center2 = np.mean(arr2_norm, axis=0)
    center2_norm = center2 / (np.linalg.norm(center2) + 1e-8)

    # Step 3: compute similarity to center
    sims_to_center2 = np.dot(arr2_norm, center2_norm)  # shape: (A,)

    # Step 4: select top-K% closest to center
    B = arr2.shape[0]
    top_k = max(1, int(B * 1))
    top_indices = np.argsort(sims_to_center2)[-top_k:]

    # Step 5: filter arr1 by selected indices
    arr2_filtered = arr2[top_indices]



    arr1_filtered_norm = arr1_filtered / (np.linalg.norm(arr1_filtered, axis=1, keepdims=True) + 1e-8)
    arr2_filtered_norm = arr2_filtered / (np.linalg.norm(arr2_filtered, axis=1, keepdims=True) + 1e-8)

    # Step 7: compute cosine similarity matrix
    sim_matrix = np.dot(arr1_filtered_norm, arr2_filtered_norm.T)  # shape: (filtered_A, B)
    # sim_matrix = sim_matrix / 0.1

    k = int(sim_matrix.shape[1] * 0.1)  # 比如 top 20%
    topk_vals = np.sort(sim_matrix, axis=1)[:, -2:]
    score = np.mean(topk_vals)


    # Step 8: for each row in filtered arr1, take max sim with arr2
    row_max_sim = np.max(sim_matrix, axis=1)  # shape: (filtered_A,)
    row_max_2 = np.max(sim_matrix, axis=0)
    # return score
    #return np.max(row_max_sim)
    return np.mean(row_max_sim)
    #return np.dot(center_norm, center2_norm)
## 中心表征 相似度
# def pairwise_cosine_similarity_score(arr1, arr2,keep_percent=1):
#
#     ## 余弦相似度距离
#     # mean1  = np.mean(arr1,axis=0)
#     # mean2 = np.mean(arr2,axis=0)
#     # mean1_norm = mean1 / (np.linalg.norm(mean1) + 1e-8)
#     # mean2_norm = mean2 / (np.linalg.norm(mean2) + 1e-8)
#     # cos_sim = np.dot(mean1_norm, mean2_norm.T)
#     # return cos_sim
#     ## 点积（不归一化）
#     # mean1 = np.mean(arr1, axis=0)
#     # mean2 = np.mean(arr2, axis=0)
#     # return np.dot(mean1, mean2)
#
#     ##曼哈顿距离（L1距离）
#     # mean1 = np.mean(arr1, axis=0)
#     # mean2 = np.mean(arr2, axis=0)
#     # return np.sum(np.abs(mean1 - mean2))
#
#     ## 高斯距离
#     # mean1 = np.mean(arr1, axis=0)
#     # mean2 = np.mean(arr2, axis=0)
#     # dist_sq = np.sum((mean1 - mean2) ** 2)
#     # return np.exp(-dist_sq / (2 * 1 ** 2))
#
#     ## 欧式距离
#     center = np.mean(arr2, axis=0)
#
#     # Step 3: compute similarity to center
#     sims_to_center = np.dot(arr2, center)  # shape: (A,)
#
#     # Step 4: select top-K% closest to center
#     A = arr2.shape[0]
#     top_k = max(1, int(A * keep_percent))
#     top_indices = np.argsort(sims_to_center)[-top_k:]
#
#
#     center1 = np.mean(arr1, axis=0)
#
#     # Step 3: compute similarity to center
#     sims_to_center1 = np.dot(arr1, center1)  # shape: (A,)
#
#     # Step 4: select top-K% closest to center
#     B = arr1.shape[0]
#     top_k = max(1, int(B * 1))
#     top_indices1 = np.argsort(sims_to_center1)[-top_k:]
#
#     # Step 5: filter arr1 by selected indices
#     arr2_filtered = arr2[top_indices]
#     arr1_filtered = arr1[top_indices1]
#     mean1 = np.mean(arr1_filtered, axis=0)
#     mean2 = np.mean(arr2_filtered, axis=0)
#
#     # A = arr1.shape[0]
#     # B = arr2.shape[0]
#
#     weight = 1+0.01*np.log(1+B)
#
#     mean1_norm = mean1 / (np.linalg.norm(mean1) + 1e-8)
#     mean2_norm = mean2 / (np.linalg.norm(mean2) + 1e-8)
#     cos_sim = np.dot(mean1_norm, mean2_norm.T)
#
#     # mean1 = np.mean(arr1, axis=0)
#     # mean2 = np.mean(arr2, axis=0)
#
#     # # L2 距离
#     dist = np.linalg.norm(mean1_norm - mean2_norm)
#
#     # 转化为相似度（越大越相似）
#     sim = 1 / (1 + dist)
#     return sim
#

# def pairwise_cosine_similarity_score(arr1, arr2, top_percent=0.3):
#         # Normalize rows to unit vectors
#     arr1_norm = arr1 / (np.linalg.norm(arr1, axis=1, keepdims=True) + 1e-8)
#     arr2_norm = arr2 / (np.linalg.norm(arr2, axis=1, keepdims=True) + 1e-8)

#     # Compute cosine similarity matrix (A, B)
#     sim_matrix = np.dot(arr1_norm, arr2_norm.T)

#     # Determine top-k count per row based on percentage
#     B = sim_matrix.shape[1]
#     top_k = max(1, math.ceil(B * top_percent))  # Ensure at least 1

#     # Get top-k similarities per row
#     topk_per_row = np.sort(sim_matrix, axis=1)[:, -top_k:]  # shape: (A, top_k)
#     row_topk_means = np.mean(topk_per_row, axis=1)           # shape: (A,)

#     # Final: average across all rows
#     return np.mean(row_topk_means)

# instance-instance 相似度最大
# def pairwise_cosine_similarity_score(arr1, arr2):
#     # Normalize
#     arr1_norm = arr1 / (np.linalg.norm(arr1, axis=1, keepdims=True) + 1e-8)
#     arr2_norm = arr2 / (np.linalg.norm(arr2, axis=1, keepdims=True) + 1e-8)
#
#     # (A, B)
#     sim_matrix = np.dot(arr1_norm, arr2_norm.T)
#
#     # 每一行：与 arr2 所有样本的平均相似度
#     row_mean_sim = np.mean(sim_matrix, axis=1)
#
#     # 再对 arr1 所有样本取平均
#     return np.mean(row_mean_sim)
# def pairwise_cosine_similarity_score(arr1, arr2):
#     # Normalize each row to unit vector
#     arr1_norm = arr1 / (np.linalg.norm(arr1, axis=1, keepdims=True) + 1e-8)
#     arr2_norm = arr2 / (np.linalg.norm(arr2, axis=1, keepdims=True) + 1e-8)
#
#     # Compute cosine similarity matrix of shape (A, B)
#     sim_matrix = np.dot(arr1_norm, arr2_norm.T)
#
#     A = arr1.shape[0]
#     B = arr2.shape[0]
#     K = int(min(A, B))  # 修正K：不能比列数还大
#
#     # 获取每一行 top-K 最大相似度（降序排列后取前K个）
#     topk_sim = np.partition(sim_matrix, -K, axis=1)[:, -K:]  # (A, K)
#
#     # 平均所有 top-K 相似度
#     return np.mean(topk_sim)
## 二范数距离
# def pairwise_cosine_similarity_score(arr1, arr2):
#     A = arr1.shape[0]
#     B = arr2.shape[0]
#
#     # 扩展维度计算所有 pairwise 的欧氏距离 (A, B)
#     arr1_expand = np.expand_dims(arr1, axis=1)  # (A, 1, K)
#     arr2_expand = np.expand_dims(arr2, axis=0)  # (1, B, K)
#
#     # L2 距离矩阵
#     dist_matrix = np.linalg.norm(arr1_expand - arr2_expand, axis=2)  # (A, B)
#
#     # 相似度 = 1 / (1 + distance)，越近相似度越高
#     sim_matrix = 1 / (1 + dist_matrix)
#
#     # 每一行选择最相似（即最小距离 → 最大相似度）
#     row_max_sim = np.max(sim_matrix, axis=1)
#
#     return np.mean(row_max_sim)
## instance-instance similarity
# def pairwise_cosine_similarity_score(arr1, arr2):
#     # arr1: (A, K), arr2: (B, K)
#     # Normalize each row to unit vector
#     arr1_norm = arr1 / (np.linalg.norm(arr1, axis=1, keepdims=True) + 1e-8)
#     arr2_norm = arr2 / (np.linalg.norm(arr2, axis=1, keepdims=True) + 1e-8)
    
#     # Compute cosine similarity matrix: (A, B)
#     sim_matrix = np.dot(arr1_norm, arr2_norm.T)
    
#     # Average all pairwise similarities
#     return np.mean(sim_matrix)


def compute_all_pairwise_similarities(arr_list):
    n = len(arr_list)
    sim_list = []

    for i, j in combinations(range(n), 2):
        sim = pairwise_cosine_similarity_score(arr_list[i], arr_list[j])
        sim_list.append(((i, j), sim))

    return sim_list
# def ClusterMerge(clusterResult, all_idxes, all_feats, match_result,match_result_unlabeled,concept_type_path,ins_type_path,id2type,types,unknown_cluster_filter,unknown_id_label):
#     cluster_ins_re_dict = clusterResult['cluster_ins_re_dict']
#
#     ## 将known cluster的type path填充好，然后将所有unknown cluster的type path设为-1
#
#     match_dict=defaultdict()
#     for pair in match_result:
#         c_idx = pair[0]
#         i_idx=pair[1]
#         level_1_concept_idx=concept_type_path[c_idx][0]
#         level_1_inscluster_idx = ins_type_path[i_idx][0]
#
#         match_dict[level_1_inscluster_idx]=concept_type_path[c_idx]
#
#     match_dict_gold=defaultdict()
#     for pair in match_result_unlabeled:
#         c_idx = pair[0]
#         i_idx=pair[1]
#         level_1_concept_idx=concept_type_path[c_idx][0]
#         level_1_inscluster_idx = ins_type_path[i_idx][0]
#
#         match_dict_gold[level_1_inscluster_idx]=concept_type_path[c_idx]
#
#
#     # for k in match_dict.keys():
#     #     if k not in match_dict_gold.keys():
#     #         print(k)
#
#     # unknown_cluster_idxs=[]
#     # known_cluster_idxs=[]
#     level_num = len(concept_type_path[0])
#     #
#     # for i_p in ins_type_path:
#     #     level_1_inscluster_idx = i_p[0]
#     #     if level_1_inscluster_idx not in match_dict.keys():
#     #         unknown_cluster_idxs.append(level_1_inscluster_idx)
#     #     else:
#     #         known_match_concept_path = match_dict[level_1_inscluster_idx]
#     #         known_cluster_idxs.append(known_match_concept_path)
#     refine_ins_type_path=copy.deepcopy(ins_type_path)
#     for j in range(len(refine_ins_type_path)):
#         re_ins_path = refine_ins_type_path[j]
#         level_1_inscluster_idx = re_ins_path[0]
#         if level_1_inscluster_idx not in match_dict.keys():
#             refine_ins_type_path[j][1:] = [-1]*(level_num-1)
#         else:
#             refine_ins_type_path[j][1:] =  copy.deepcopy(match_dict[level_1_inscluster_idx][1:])
#
#     gold_ins_type_path=copy.deepcopy(ins_type_path)
#     # for i in range(1,level_num):
#
#     for j in range(len(gold_ins_type_path)):
#         re_ins_path = gold_ins_type_path[j]
#         level_1_inscluster_idx = re_ins_path[0]
#         if level_1_inscluster_idx not in match_dict_gold.keys():
#             gold_ins_type_path[j][1:] = [-1]*(level_num-1)
#         else:
#             gold_ins_type_path[j][1:] =  copy.deepcopy(match_dict_gold[level_1_inscluster_idx][1:])
#
#     ## 根据上面的type path，将parent-child的node 中的instance表征集合构建出来，这个地方的话，我觉得还是只把known type 的层次聚类树构建出来，unknown的单独构成一个list
#     ##这样可以方便后面去遍历
#     parent_child_dict = {i: {} for i in range(1, level_num+1)}
#     parent_child_rep_dict={i: {} for i in range(1, level_num+1)}
#     all_level_unknown_sim_list=[]
#     unknwn_cluster_rep = []
#     for i in range(1, level_num):
#         for i_p in refine_ins_type_path:
#             cur_level_idx = i_p[i]
#             pre_level_idx = i_p[i - 1]
#             if cur_level_idx != -1:
#                 if cur_level_idx not in parent_child_dict[i].keys():
#                     parent_child_dict[i][cur_level_idx] = [pre_level_idx]
#                     if i == 1:
#                         parent_child_rep_dict[i][cur_level_idx] = copy.deepcopy(cluster_ins_re_dict[pre_level_idx])
#                     else:
#                         parent_child_rep_dict[i][cur_level_idx]=copy.deepcopy(parent_child_rep_dict[i-1][pre_level_idx])
#                 else:
#                     #if pre_level_idx not in parent_child_dict[i][cur_level_idx]:
#                     parent_child_dict[i][cur_level_idx].append(pre_level_idx)
#                     if i == 1:
#                         parent_child_rep_dict[i][cur_level_idx] = np.concatenate([parent_child_rep_dict[i][cur_level_idx],copy.deepcopy(cluster_ins_re_dict[pre_level_idx])],axis=0)
#                     else:
#                         parent_child_rep_dict[i][cur_level_idx]=np.concatenate([parent_child_rep_dict[i][cur_level_idx],copy.deepcopy(parent_child_rep_dict[i-1][pre_level_idx])],axis=0)
#
#     ## 按道理来说，我们应该把最底层的
#     parent_child_rep_dict_gold={i: {} for i in range(0, level_num+1)}
#     for i in range(1, level_num):
#         for i_p in gold_ins_type_path:
#             cur_level_idx = i_p[i]
#             pre_level_idx = i_p[i - 1]
#             if cur_level_idx != -1:
#                 if cur_level_idx not in parent_child_rep_dict_gold[i].keys():
#                     if i == 1:
#                         parent_child_rep_dict_gold[i][cur_level_idx] = copy.deepcopy(cluster_ins_re_dict[pre_level_idx])
#                     else:
#                         parent_child_rep_dict_gold[i][cur_level_idx]=copy.deepcopy(parent_child_rep_dict_gold[i-1][pre_level_idx])
#                 else:
#                     if i == 1:
#                         parent_child_rep_dict_gold[i][cur_level_idx] = np.concatenate([parent_child_rep_dict_gold[i][cur_level_idx],copy.deepcopy(cluster_ins_re_dict[pre_level_idx])],axis=0)
#                     else:
#                         parent_child_rep_dict_gold[i][cur_level_idx]=np.concatenate([parent_child_rep_dict_gold[i][cur_level_idx],copy.deepcopy(parent_child_rep_dict_gold[i-1][pre_level_idx])],axis=0)
#
#
#     ##这地方用来计算gold_parent_child_dict,从而作为下面的评测标准
#     parent_child_dict_gold = {i: {} for i in range(1, level_num+1)}
#     for i in range(1, level_num):
#         for i_p in gold_ins_type_path:
#             cur_level_idx = i_p[i]
#             pre_level_idx = i_p[i - 1]
#             if cur_level_idx != -1:
#                 if cur_level_idx not in parent_child_dict_gold[i].keys():
#                     parent_child_dict_gold[i][cur_level_idx] = [pre_level_idx]
#                 else:
#                     parent_child_dict_gold[i][cur_level_idx].append(pre_level_idx)
#
#     ## 这地方单独把root表征计算出来,以及吧root下的子节点也构建出来
#     parent_child_rep_dict[level_num][0]=[]
#     parent_child_dict[level_num][0] = []
#     for k,v in parent_child_rep_dict[level_num-1].items():
#         if parent_child_rep_dict[level_num][0] == []:
#             parent_child_rep_dict[level_num][0] = copy.deepcopy(v)
#         else:
#             parent_child_rep_dict[level_num][0] =np.concatenate([parent_child_rep_dict[level_num][0],copy.deepcopy(v)],axis=0)
#         parent_child_dict[level_num][0].append(k)
#
#     ##对gold_parent_child_dict的root进行构建
#     parent_child_dict_gold[level_num][0] = []
#     for k,v in parent_child_dict_gold[level_num-1].items():
#         parent_child_dict_gold[level_num][0].append(k)
#
#     ## 这地方单独把gold root表征计算出来
#     parent_child_rep_dict_gold[level_num][0]=[]
#     for k,v in parent_child_rep_dict_gold[level_num-1].items():
#         if parent_child_rep_dict_gold[level_num][0] == []:
#             parent_child_rep_dict_gold[level_num][0] = copy.deepcopy(v)
#         else:
#             parent_child_rep_dict_gold[level_num][0] =np.concatenate([parent_child_rep_dict_gold[level_num][0],copy.deepcopy(v)],axis=0)
#
#     ## 这地方吧gold的最底层的cluster表征也放上
#     for i_p in gold_ins_type_path:
#         cur_level_idx = i_p[0]
#         parent_level_idx = i_p[1]
#         if parent_level_idx != -1:
#             parent_child_rep_dict_gold[0][cur_level_idx] = copy.deepcopy(cluster_ins_re_dict[cur_level_idx])
#
#
#     ##获取所有的unknown cluster list以及所有的unknown cluster id
#     unknown_cluster_idxs = []
#     for i_p in refine_ins_type_path:
#         cur_level_idx = i_p[1]
#         pre_level_idx = i_p[0]
#         if cur_level_idx == -1:
#             unknwn_cluster_rep.append(cluster_ins_re_dict[pre_level_idx])
#             unknown_cluster_idxs.append(pre_level_idx)
#         # parent_child_rep_dict_temp[i] = copy.deepcopy(parent_child_rep_dict[i])
#         # for k,v in parent_child_rep_dict[i].items():
#         #     if k == -1 and i == 1:
#         #         unknwn_cluster_rep = copy.deepcopy(parent_child_rep_dict[i][-1])
#         #     parent_child_sim_dict[i][k] = compute_all_pairwise_similarities(parent_child_rep_dict[i][k])
#         #     parent_child_rep_dict[i][k] = np.concatenate(parent_child_rep_dict[i][k],axis=0)
#
#
#     ## 我们在这里把unknown cluster与所有的known type的相似度计算出来
#     unknown_sim = [[] for j in range(len(unknwn_cluster_rep)) ]
#     for j in range(len(unknown_sim)):
#         if unknown_cluster_idxs[j] in unknown_id_label.keys():
#             for i in range(level_num,0,-1):
#                     for k, v in parent_child_rep_dict[i].items():
#                         if i == level_num:
#                             type = 'root'
#                         else:
#                             type = id2type[k]
#                         unknown_sim[j].append( (type,pairwise_cosine_similarity_score(unknwn_cluster_rep[j],parent_child_rep_dict[i][k])))
#
#     for j in range(len(unknown_sim)):
#         print(unknown_sim[j])
#     ## 对parent_child_dict_gold与parent_child_dict进行评测
#     gold_unknown = [[] for j in range(len(unknwn_cluster_rep)) ]
#     for j in range(len(unknwn_cluster_rep)):
#         for re_ins_path in gold_ins_type_path:
#             if unknown_cluster_idxs[j] == re_ins_path[0]:
#                 if re_ins_path[1] != -1:
#                     for i in range(len(re_ins_path)):
#                         if i == 0:
#                             id_type = id2type[match_dict_gold[re_ins_path[i]][0]]
#                         else:
#                             id_type = id2type[re_ins_path[i]]
#                         gold_unknown[j].append(id_type)
#                 else:
#                         gold_unknown[j]=copy.deepcopy(re_ins_path[0:])
#
#     for j in range(len(gold_unknown)):
#         if gold_unknown[j][1] != -1:
#             gold_unknown[j].append('root')
#
#
#     ## 将unknown clutster与一层一层的known cluster表征进行搜索
#     unknown_predict=[[] for j in range(len(unknwn_cluster_rep)) ]
#
#     unknown_length = [unknwn_cluster_rep[j].shape[0] for j in range(len(unknwn_cluster_rep))]
#     unknown_sort_index_list = np.argsort(-np.array(unknown_length))
#     for j in unknown_sort_index_list:
#     #for j in range(len(unknwn_cluster_rep)):
#         if unknown_cluster_idxs[j] not in unknown_cluster_filter:
#             continue
#         sim_max = -100
#         best_level = level_num-1
#         best_node=0
#         search_flag=False
#         for i in range(level_num,0,-1):
#             search_flag=False
#             if i == level_num:
#                 child_idxs=[0]
#             else:
#                 child_idxs = parent_child_dict[i+1][best_node]
#             for k, v in parent_child_rep_dict[i].items():
#                 if k in child_idxs:
#                     sim=pairwise_cosine_similarity_score(unknwn_cluster_rep[j],parent_child_rep_dict[i][k])
#                     if sim> sim_max:
#                         # if i == level_num or sim > 0.25:
#                             sim_max = sim
#                             #继续搜索
#                             search_flag=True
#                             best_level = i
#                             best_node = k
#
#             if search_flag:
#                 unknown_predict[j].append(best_node)
#             if not search_flag:
#                 break
#
#         if best_level == level_num:
#             print('best level')
#             print(unknown_cluster_idxs[j])
#         for i in range(level_num,level_num-len(unknown_predict[j]),-1):
#             node = unknown_predict[j][level_num-i]
#             parent_child_rep_dict[i][node] = np.concatenate([parent_child_rep_dict[i][node],unknwn_cluster_rep[j]],axis=0)
#             parent_child_dict[i][node].append(unknown_cluster_idxs[j])
#         # if best_level == level_num-1:
#         #     print(unknown_cluster_idxs[j])
#     for j in range(len(unknown_predict)):
#         if unknown_cluster_idxs[j] not in unknown_cluster_filter:
#             continue
#         for i in range(len(unknown_predict[j])):
#             if i == 0:
#                 unknown_predict[j][i] = 'root'
#             else:
#                 unknown_predict[j][i] = id2type[unknown_predict[j][i]]
#
#
#     ##这地方或者可以先判断一下parent_child_dict_gold本身的情况
#     # for i in range(1, level_num):
#     #     for k,v in parent_child_rep_dict_gold:
#
#
#
#
#
#
#     ## 对match的位置进行评测
#     # match_rate= 0
#     # for j in range(len(unknwn_cluster_rep)):
#
#     #     match_rate += len(list(set(unknown_id_label[unknown_cluster_idxs[j]]).intersection(unknown_predict[j])))/len(unknown_predict[j])
#     # match_rate = match_rate / len(unknwn_cluster_rep)
#     # print(match_rate)
#
#     ## 判断gold中应该插入的cluster的匹配程度
#     match_rate= 0
#     num=0
#     for j in range(len(unknwn_cluster_rep)):
#         if unknown_cluster_idxs[j] in unknown_id_label.keys():
#             # if gold_unknown[j][1] != -1 and gold_unknown[j][0] not in types:
#                 rate_item = len(list(set(unknown_id_label[unknown_cluster_idxs[j]]).intersection(unknown_predict[j])))/len(unknown_predict[j])
#                 match_rate += rate_item
#                 # if rate_item >0:
#                 print(unknown_cluster_idxs[j])
#                 print(unknown_id_label[unknown_cluster_idxs[j]])
#                 print(unknown_predict[j])
#                 print(rate_item)
#                 print("...........................")
#                 num += 1
#     match_rate = match_rate / num
#     print(match_rate)
#     print(num)













    # unknown_sim_rank_dict = {k: [] for k in parent_child_dict[1][-1]}
    # all_level_unknown_sim_list = []
    # for i in range(1, level_num):
    #     if -1 in parent_child_dict[i].keys():
    #         unknown_sim_dict = {k: {} for k in parent_child_dict[i][-1]}

    #         for j in range(len(unknwn_cluster_rep)):
    #             for k, v in parent_child_rep_dict[i].items():
    #                 if k != -1:
    #                     if k not in id2type.keys():
    #                         type_name = 'unknown'+str(k)
    #                         id2type[k] = type_name
    #                     unknown_sim_dict[parent_child_dict[i][-1][j]][id2type[k]] = pairwise_cosine_similarity_score(
    #                         unknwn_cluster_rep[j],
    #                         parent_child_rep_dict[i][k])
    #                     child_rep = parent_child_rep_dict_temp[i][k]
    #                     childs = parent_child_dict[i][k]
    #                     for c in range(len(childs)):
    #                         if childs[c] in match_dict.keys():
    #                             sim = pairwise_cosine_similarity_score(
    #                                 unknwn_cluster_rep[j], child_rep[c])
    #                             unknown_sim_dict[parent_child_dict[i][-1][j]][
    #                                 id2type[match_dict[childs[c]][i - 1]]] = sim
    #                             if i == 1:
    #                                 unknown_sim_rank_dict[parent_child_dict[i][-1][j]].append(
    #                                         (id2type[match_dict[childs[c]][i - 1]], sim))
    #                         if i ==1:
    #                             unknown_sim_rank_dict[parent_child_dict[i][-1][j]] = sorted(
    #                                 unknown_sim_rank_dict[parent_child_dict[i][-1][j]], key=lambda x: x[1], reverse=True)

    #         all_level_unknown_sim_list.append(unknown_sim_dict)
    #     else:
    #         all_level_unknown_sim_list.append(None)

import copy
import numpy as np
from collections import defaultdict


# ---------- 工具函数 ----------

def build_cluster_map(match_result, concept_type_path, ins_type_path):
    """构建 cluster → concept path 映射"""
    cluster_map = {}
    for c_idx, i_idx in match_result:
        cluster_id = ins_type_path[i_idx][0]
        cluster_map[cluster_id] = concept_type_path[c_idx]
    return cluster_map


def refine_type_path(ins_type_path, cluster_map, level_num):
    """refine type path：unknown → -1"""
    refined = copy.deepcopy(ins_type_path)

    for path in refined:
        cluster_id = path[0]
        path[0] = 100 + path[0]
        if cluster_id not in cluster_map:

            path[1:] = [-1] * (level_num - 1)
        else:
            path[1:] = cluster_map[cluster_id][1:]
    return refined


def build_parent_child_structure(type_paths, cluster_ins_re_dict, cluster_ins_data_dict,level_num):
    """
    构建：
    - parent_child_dict
    - parent_child_rep_dict（embedding aggregation）
    """
    parent_child_dict = {i: {} for i in range(1, level_num + 1)}
    parent_child_rep_dict = {i: {} for i in range(1, level_num + 1)}
    parent_child_data_dict = {i: {} for i in range(1, level_num + 1)}

    for level in range(1, level_num):
        for path in type_paths:
            cur = path[level]
            prev = path[level - 1]

            if cur == -1:
                continue

            if cur not in parent_child_dict[level]:
                parent_child_dict[level][cur] = [prev]

                if level == 1:
                    parent_child_rep_dict[level][cur] = copy.deepcopy(
                        cluster_ins_re_dict[prev-100]
                    )
                    parent_child_data_dict[level][cur] = copy.deepcopy(
                        cluster_ins_data_dict[prev - 100]
                    )
                else:
                    parent_child_rep_dict[level][cur] = copy.deepcopy(
                        parent_child_rep_dict[level - 1][prev]
                    )
                    parent_child_data_dict[level][cur] = copy.deepcopy(
                        parent_child_data_dict[level - 1][prev]
                    )
            else:
                if prev not in parent_child_dict[level][cur]:
                    parent_child_dict[level][cur].append(prev)

                if level == 1:
                    parent_child_rep_dict[level][cur] = np.concatenate(
                        [
                            parent_child_rep_dict[level][cur],
                            cluster_ins_re_dict[prev-100]
                        ],
                        axis=0
                    )
                    parent_child_data_dict[level][cur] = np.concatenate(
                        [
                            parent_child_data_dict[level][cur],
                            cluster_ins_data_dict[prev - 100]
                        ],
                        axis=0
                    )
                else:
                    parent_child_rep_dict[level][cur] = np.concatenate(
                        [
                            parent_child_rep_dict[level][cur],
                            parent_child_rep_dict[level - 1][prev]
                        ],
                        axis=0
                    )
                    parent_child_data_dict[level][cur] = np.concatenate(
                        [
                            parent_child_data_dict[level][cur],
                            parent_child_data_dict[level - 1][prev]
                        ],
                        axis=0
                    )

    return parent_child_dict, parent_child_rep_dict,parent_child_data_dict


def build_root_node(parent_child_dict, parent_child_rep_dict, parent_child_data_dict, level_num):
    """构建 root 层"""
    parent_child_dict[level_num][999] = []
    parent_child_rep_dict[level_num][999] = []
    parent_child_data_dict[level_num][999] = []

    for k, v in parent_child_rep_dict[level_num - 1].items():
        if len(parent_child_rep_dict[level_num][999]) == 0:
            parent_child_rep_dict[level_num][999] = copy.deepcopy(v)
            parent_child_data_dict[level_num][999] = copy.deepcopy(parent_child_data_dict[level_num - 1][k])
        else:
            parent_child_rep_dict[level_num][999] = np.concatenate(
                [parent_child_rep_dict[level_num][999], v], axis=0
            )
            parent_child_data_dict[level_num][999] = np.concatenate(
                [parent_child_data_dict[level_num][999], parent_child_data_dict[level_num - 1][k]], axis=0
            )
        parent_child_dict[level_num][999].append(k)


def extract_unknown_clusters(refined_paths, cluster_ins_re_dict,cluster_ins_data_dict):
    """提取 unknown clusters"""
    unknown_ids = []
    unknown_reps = []
    unknown_data=[]

    for path in refined_paths:
        if path[1] == -1:
            cluster_id = path[0]
            unknown_ids.append(cluster_id)
            unknown_reps.append(cluster_ins_re_dict[cluster_id-100])
            unknown_data.append(cluster_ins_data_dict[cluster_id-100])

    return unknown_ids, unknown_reps,unknown_data


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_representative_samples(cluster_rep, cluster_data, num_samples=20):
    """
    从 cluster 中选出最接近中心的代表性样本（返回原始数据）

    Args:
        cluster_rep: np.ndarray, shape (N, D)
        cluster_data: list, 长度 N，对应原始数据
        num_samples: int

    Returns:
        representative_samples: list
    """

    assert len(cluster_rep) == len(cluster_data), "rep 和 data 数量不一致"

    # Step 1: 计算 cluster center（centroid）
    cluster_center = np.mean(cluster_rep, axis=0, keepdims=True)  # shape (1, D)

    # Step 2: 计算所有样本与中心的 cosine similarity
    sims = cosine_similarity(cluster_rep, cluster_center).flatten()  # shape (N,)

    # Step 3: 取相似度最高的 top-k
    topk_indices = np.argsort(sims)[-num_samples:][::-1]

    # Step 4: 返回对应的原始数据
    representative_samples = [cluster_data[i] for i in topk_indices]

    return representative_samples


def generate_summary(samples,type_desc,parent_type,parent_to_children):
    """
    Use a large language model to generate a semantic summary for a set of samples.

    Args:
    - samples (list): A list of sample data (e.g., feature vectors or text).

    Returns:
    - str: A semantic summary generated by the LLM.
    """
    siblings = parent_to_children[parent_type]
    known_types_prompt = "\n".join([f"{type}:{desc}" for type,desc in type_desc.items()])
    mention_sentence = ""

    for sample in samples:
        tokens = sample["sentence"]
        sentence = ' '.join(tokens)
        start = sample['mention']['start']
        end = sample['mention']['end']
        mention = ' '.join(tokens[start:end])
        mention_sentence += (
            f"Sentence: {sentence}\n"
            f"Mention: {mention}\n\n"
        )

    prompt = [{
        "role": "system",
        "content":
            "You are an AI assistant specializing in entity typing and taxonomic modeling.\n"
            "Your task is to infer a new unknown type based on the provided samples, "
            "and generate BOTH a semantically precise type name and a high-quality description.\n\n"
            
            
            f"Parent type:\n{parent_type}\n\n"
            
            "Existing sibling types:\n"
            f"{siblings}\n\n"
            
            
            
            "Requirements:\n"
            "• Generate ONLY ONE child type name.\n"
            "• The child type name must represent the shared semantic characteristics of the samples.\n"
            "• The child type must be semantically distinct from existing sibling types.\n"
            "• Do NOT generate a synonym, paraphrase, or minor variation of an existing sibling type.\n"
            "• Prefer the most specific valid abstraction supported by the samples.\n"
            "• Use lowercase and underscores only.\n"
            "• The child type name itself MUST NOT contain '/'.\n"
            "• The child type name should be concise and taxonomy-friendly.\n\n"


            "Description requirements:\n"
            "For the generated type, write a structured paragraph (3–4 sentences) that:\n"
            "• Defines the core semantic scope of the type in relation to its parent;\n"
            "• Clearly reflects and explains the meaning of the generated type name;\n"
            "• Highlights its defining characteristics (without explicit comparison wording);\n"
            "• Maintains consistency in tone and abstraction level with existing type descriptions;\n"
            "• Optionally includes representative real-world examples.\n\n"

            "Alignment constraints:\n"
            "• The description MUST be semantically consistent with the type name.\n"
            "• The description MUST NOT collapse into any existing known type.\n"
            "• The type must introduce a clear semantic boundary in the taxonomy.\n\n"

            "Known types and descriptions:\n"
            f"{known_types_prompt}\n\n"

            "Samples:\n"
            f"{mention_sentence}\n\n"

            "Output format (strict):\n"
            "Child Type Name: <child_name_only>\n"
            "Description: <description>\n"
    }]
    # prompt = [{"role":"system",
    #     "content":
    #         f"You are an AI assistant specializing in fine-grained entity typing.\n"
    #         f"Your task is to summarize the descriptions of key type semantic meaning of these samples for some given mention within a cluster.\n\n"}]
    key = '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'
    model = OpenAI(api_key=key, base_url='https://openrouter.ai/api/v1')
    response = model.chat.completions.create(
        model='gpt-5.4',
        # engine="text-davinci-003",  # Using GPT model
        messages=prompt,
        # max_tokens=150,  # Generate a reasonable length summary
        temperature=0, # Control the randomness of the output
        seed=0
    )
    text = response.choices[0].message.content.strip()
    # =========================
    # Parse output
    # =========================
    child_name = None
    description = None

    child_match = re.search(
        r"Child Type Name:\s*(.+?)(?:\n|$)",
        text
    )

    desc_match = re.search(
        r"Description:\s*([\s\S]+)",
        text
    )

    if child_match:
        child_name = child_match.group(1).strip()

    if desc_match:
        description = desc_match.group(1).strip()

    # =========================
    # Fallback parsing
    # =========================
    if child_name is None or description is None:

        lines = text.split("\n")

        if len(lines) >= 2:
            child_name = (
                lines[0]
                .replace("Child Type Name:", "")
                .strip()
            )

            description = (
                "\n".join(lines[1:])
                .replace("Description:", "")
                .strip()
            )

    # =========================
    # Normalize child name
    # =========================
    child_name = child_name.lower()

    child_name = child_name.replace(" ", "_")

    # 防止模型偷偷生成 path
    child_name = child_name.split("/")[-1]

    # =========================
    # Construct full type path
    # =========================
    full_type = f"{parent_type}/{child_name}"

    return full_type, description


def llm_decision_function(top_k_nodes, cluster_rep, cluster_data,
                          parent_child_rep_dict, parent_child_data_dict, level, type_desc,type2id,decision_type="option_1"):
    """
    Use a large language model (LLM) to determine which of the top-k nodes best matches
    the current cluster's semantics based on semantic summaries.

    decision_type:
        "option_1" - Compare cluster's representative samples with node's semantic summary
        "option_2" - Compare cluster's semantic summary with node's semantic summary
    """

    # Step 1: Get the representative samples for the current cluster
    representative_cluster_samples = extract_representative_samples(cluster_rep, cluster_data)

    # Step 2: Generate the semantic summary for the current cluster using the representative samples
    # type_name, description = generate_summary(representative_cluster_samples,type_desc)

    mention_sentence = ""

    for sample in representative_cluster_samples:
        tokens = sample["sentence"]
        sentence = ' '.join(tokens)
        start = sample['mention']['start']
        end = sample['mention']['end']
        mention = ' '.join(tokens[start:end])
        mention_sentence += (
            f"Sentence: {sentence}\n"
            f"Mention: {mention}\n\n"
        )
    all_sample_type = []
    # Step 4: Construct the prompt based on the decision type
    if decision_type == "option_1":
        # Option 1: Compare cluster representative samples with node semantic summaries
        prompt = ("Given the following cluster representative samples and candidate type description, which type is most semantically similar to the current cluster?\n\n"
                
                "Constraints:\n"
                "- Only choose from the provided candidate types\n"
                "- Final answer must be one of the candidate type names\n\n"
                
                "Output requirements:\n"
                "- Return ONLY one type name\n"
                "- Do NOT provide explanation or extra text\n\n"
        #         "- Return ONLY one type name\n"
        #         "- Do NOT provide explanation or extra text\n\n"
                f"Current cluster representative samples:\n {mention_sentence}\n\n"
                "Candidate type description:\n")

        # for sample in representative_cluster_samples:
        #     sentence = ' '.join(sample["sentence"])
        #     start = sample['mention']['start']
        #     end = sample['mention']['end']
        #     mention = sentence[start:end]
        #     prompt = (
        #         "You are an AI assistant specializing in entity typing.\n"
        #         "Your task is to select the type that is most semantically similar to the given mention in a sentence.\n\n"
        #
        #         "Constraints:\n"
        #         "- Each mention must be assigned to exactly ONE type\n"
        #         "- Only choose from the provided candidate types\n"
        #         "- Final answer must be one of the candidate type names\n\n"
        #
        #         "Output requirements:\n"
        #         "- Return ONLY one type name\n"
        #         "- Do NOT provide explanation or extra text\n\n"
        #
        #         "Candidate types and description:\n"
        #     )
        for type,desc in type_desc.items():
            if type in top_k_nodes:
                prompt += f"{type}:{desc}\n"
            # prompt+=f"Input:\nSentence: {sentence}\nMention: {mention}\n\n"
        messages = [{
                "role": "system",
                "content": prompt
        }]
            # Step 5: Call the large language model (e.g., OpenAI GPT)
        key = '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'
        model = OpenAI(api_key=key, base_url='https://openrouter.ai/api/v1')
        response = model.chat.completions.create(
                model='gpt-5.4',
                # engine="text-davinci-003",  # Using GPT model
                messages=messages,
                # max_tokens=150,  # Generate a reasonable length summary
                temperature=0,  # Control the randomness of the output
                seed=0
        )
            # 我们好像可以直接让大模型将这些instance分配到这两个type中，看他们会分到哪里
            # Parse the LLM's output, assuming the model will return the best node's ID
        res = response.choices[0].message.content.strip()
        all_sample_type.append(res)
        # best_node = max(set(all_sample_type), key=all_sample_type.count)
        best_node=res

    elif decision_type == "option_2":
        # Option 1: Compare cluster representative samples with node semantic summaries
        prompt = (
            "Given the following cluster representative samples and candidate type descriptions, "
            "which type is most semantically similar to the current cluster?\n\n"

            "Important:\n"
            "- If NONE of the candidate types are semantically suitable,\n"
            "  return 'None'\n"
            "- Only return 'None' when the semantic mismatch is clear\n\n"

            "Constraints:\n"
            "- Only choose from the provided candidate types or 'None'\n"
            "- Final answer must be exactly one candidate type name or 'None'\n\n"

            "Output requirements:\n"
            "- Return ONLY one type name or 'None'\n"
            "- Do NOT provide explanation or extra text\n\n"

            f"Current cluster representative samples:\n{mention_sentence}\n\n"
            "Candidate type descriptions:\n"
        )

        for type_name, desc in type_desc.items():
            if type_name in top_k_nodes:
                prompt += f"{type_name}: {desc}\n"

        messages = [{
            "role": "system",
            "content": prompt
        }]

        key = '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'
        model = OpenAI(
            api_key=key,
            base_url='https://openrouter.ai/api/v1'
        )

        response = model.chat.completions.create(
            model='gpt-5.4',
            messages=messages,
            temperature=0,
            seed=0
        )

        res = response.choices[0].message.content.strip()

        # 防止模型输出奇怪格式
        valid_outputs = set(top_k_nodes) | {"None"}

        if res not in valid_outputs:
            res = "None"

        all_sample_type.append(res)

        best_node = res

    # Return the node that best matches the cluster's semantics
    return best_node,all_sample_type

import re


def global_llm_parent_selection(
    cluster_rep,
    cluster_data,
    id2type,
    type_desc,
    level_num
):
    representative_cluster_samples = extract_representative_samples(cluster_rep, cluster_data)

    # ==========================================================
    # Step 2: Build candidate type list
    # ==========================================================

    candidate_types = []
    for type_name,desc in type_desc.items():
        depth = type_name.count('/')
        if depth > level_num-1:
            continue
        candidate_types.append(type_name)
    # remove duplicates
    candidate_types = list(set(candidate_types))


    # ==========================================================
    # Step 3: Build prompt
    # ==========================================================

    prompt = (
        "You are given a set of candidate parent types in a taxonomy.\n\n"

        "Your task is to select the MOST semantically suitable "
        "candidate type to serve as the parent type for the following cluster.\n\n"

        "Constraints:\n"
        "- The selected type MUST come from the candidate type list.\n"
        "- Only choose ONE candidate parent type.\n"
        "- Do NOT generate new types.\n"
        "- Do NOT provide explanations.\n"
        "- Output ONLY the selected parent type name.\n\n"

        "Candidate parent types:\n"
    )

    # ----------------------------------------------------------
    # Candidate types
    # ----------------------------------------------------------
    candidate_types.append('root')
    for type_name in candidate_types:

        # special case for root
        if type_name == 'root':
            prompt += f"- {type_name}\n"
            continue

        description = type_desc[type_name]

        prompt += (
            f"- {type_name}: {description}\n"
        )

    # ----------------------------------------------------------
    # Cluster samples
    # ----------------------------------------------------------

    prompt += "\nCluster representative samples:\n"

    for sample in representative_cluster_samples:
        tokens = sample["sentence"]
        sentence = ' '.join(tokens)
        start = sample['mention']['start']
        end = sample['mention']['end']
        mention = ' '.join(tokens[start:end])
        prompt += f'Sentence: {sentence}\nMention: {mention}\n'
    prompt += "\nAnswer:\n"

    # ==========================================================
    # Step 4: Call LLM
    # ==========================================================

    messages = [{
        "role": "system",
        "content": prompt
    }]
    # Step 5: Call the large language model (e.g., OpenAI GPT)
    key = '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'
    model = OpenAI(api_key=key, base_url='https://openrouter.ai/api/v1')
    response = model.chat.completions.create(
        model='gpt-5.4',
        # engine="text-davinci-003",  # Using GPT model
        messages=messages,
        # max_tokens=150,  # Generate a reasonable length summary
        temperature=0,  # Control the randomness of the output
        seed=0
    )
    # 我们好像可以直接让大模型将这些instance分配到这两个type中，看他们会分到哪里
    # Parse the LLM's output, assuming the model will return the best node's ID
    res = response.choices[0].message.content.strip()

    # ==========================================================
    # Step 5: Post-process output
    # ==========================================================

    answer = res

    # remove markdown formatting
    answer = re.sub(r'[`"\']', '', answer)

    # remove bullets
    answer = answer.replace("-", "").strip()

    # ==========================================================
    # Step 6: Match output to existing types
    # ==========================================================

    # lowercase exact match
    for type_name in candidate_types:
        if answer.lower() == type_name.lower():
            return type_name

    # final fallback
    return answer
def hierarchical_search(
    cluster_rep,
    cluster_data,
    parent_child_dict,
    parent_child_rep_dict,
    parent_child_data_dict,
    original_parent_child_rep_dict,
    level_num,
    id2type,
    type2id,
    unknown_id_label,
    cluster_id,
    type_desc=None,
    parent_to_children=None,
    top_k=2,  # top-k nodes for decision

):
    """逐层 greedy search"""
    best_node = 999
    best_level = level_num
    sim_max = -100

    path = []
    sims=[]
    merge_happened = False  # ⭐ 新增
    new_parent_id=None
    merge_node=None
    delay=True
    # best_node=global_llm_parent_selection(cluster_rep,cluster_data,id2type,type_desc,level_num)
    # if best_node == 'root':
    #     best_level=level_num
    #     path.append(999)
    # else:
    #     best_level=level_num-best_node.count('/')
    #     parts = best_node.strip('/').split('/')
    #
    #     path = [999]
    #     current = ''
    #
    #     # 去掉最后一个节点
    #     for p in parts:
    #         current += '/' + p
    #         path.append(type2id[current])
    # level=best_level
    # if best_node == 'root':
    #     best_node=999
    # else:
    #     if best_node.startswith('/'):
    #         best_node = type2id[best_node]
    #     else:
    #         best_node = '/'+best_node
    #         best_node = type2id[best_node]
    for level in range(level_num, 0, -1):

        if level == level_num:
            candidate_nodes = [999]
        else:
            candidate_nodes = parent_child_dict[level+1][best_node]

        found = False

        node_sim_dict = {}
        for i in range(len(candidate_nodes)):
            node = candidate_nodes[i]
            if node in id2type.keys():
                node_type=id2type[node]
            elif node==999:
                node_type='root'
            else:
                node_type=unknown_id_label[node-100][0]
            # 排除 'padding' 类型的节点
            # if 'padding' in node_type:
            #     continue  # 如果是 padding 类型，跳过该节点
            sim = pairwise_cosine_similarity_score(
                cluster_rep,
                parent_child_rep_dict[level][node]
            )
            node_sim_dict[node] = sim
            # if sim >= sim_max and 'padding' not in type:
            #     sims.append(sim_max)
            #     sim_max = sim
            #     best_node = node
            #     best_level = level
            #     found = True
            # =====================================================
            # 🔥 RE-PARENT CHECK
            # =====================================================
            # Step 2: 按相似度排序并选择 top-k
        if len(node_sim_dict)==1:
            best_node = candidate_nodes[0]
        else:
            sorted_nodes = sorted(node_sim_dict.items(), key=lambda x: x[1], reverse=True)[0:top_k]
            top_k_nodes=[]
            for node, _ in sorted_nodes:
                # if node in id2type.keys():
                top_k_nodes.append(id2type[node])
                # else:
                #
                #     representative_cluster_samples = extract_representative_samples(parent_child_rep_dict[level][node], parent_child_data_dict[level][node],num_samples=10)
                #     type_name, description = generate_summary(representative_cluster_samples, type_desc)
                #     type_desc[type_name]=description
                #     top_k_nodes.append(type_name)
                #     type2id[type_name]=node
                #     id2type[node] = type_name
            if cluster_id ==123:
                print('')
            # Step 3: LLM做决策，选择最合适的节点
            best_node,all_sample_type = llm_decision_function(top_k_nodes, cluster_rep, cluster_data,parent_child_rep_dict, parent_child_data_dict,level,type_desc,type2id,decision_type="option_2")
            # best_node, best_sim = max(node_sim_dict.items(), key=lambda x: x[1])
            # best_node = id2type[best_node]
            print('best_node:'+str(best_node))
            if not best_node.startswith('/'):
                best_node = '/'+best_node
                # best_node = type2id[best_node]
            if best_node=='None' or best_node=='/None':
                best_level = level
                break  # 如果相似度不够高，停止搜索
            if best_node not in type2id.keys():
                best_node = int(best_node.replace("unknown", ""))
            else:
                best_node=type2id[best_node]
        # Step 4: 判断是否继续搜索（embedding相似度决定）
        # sim_with_best_node = node_sim_dict.get(best_node, -100)
        # best_node, best_sim = max(node_sim_dict.items(), key=lambda x: x[1])
        # # 这说明LLM选择的node恰好是sim最大的node,在这种情况下，我们就认为，搜到头了
        # print('sim_with_best_node:'+str(sim_with_best_node))
        # print('sim_max:'+str(sim_max))
        # if best_sim<sim_max:
        #     found = True
        #     best_level = level
        #     break  # 如果相似度不够高，停止搜索

        # =====================================================
        # 🔥 merge step
        # =====================================================
        parent_level = level + 1
        ##限制只有可能node >100的时候，才有可能执行re-parent
        if best_node >100 and best_node != 999:
            ##### 如果说best node是一个unknown的node，那么其实就是这两个就该合并到一起，并且要生成一个中间的父节点，不需要进行什么判断，这没问题

            # path = path[:-1]
            new_parent_id = max(type2id.values()) + 1



            # 注册新节点
            # type_name = f"merge_{cluster_id}_{best_node}"
            # type2id[type_name] = new_parent_id
            # id2type[new_parent_id] = type_name

            # ---------- 找到当前 parent ----------
            parent_level = level + 1
            parent_node = 999 if len(path) == 0 else path[-1]
            if parent_node != 999:
                parent_type = id2type[parent_node]
            else:
                parent_type = "root"
            # ---------- 替换 parent-child 关系 ----------
            parent_child_dict[parent_level][parent_node].remove(best_node)
            parent_child_dict[parent_level][parent_node].append(new_parent_id)

            # ---------- 构建新父节点的 children ----------
            parent_child_dict[level][new_parent_id] = [best_node, cluster_id]

            # ---------- 初始化 representation ----------
            parent_child_rep_dict[level][new_parent_id] = np.concatenate(
                                                                  [parent_child_rep_dict[level][best_node] ,cluster_rep],axis=0 )

            parent_child_data_dict[level][new_parent_id] = np.concatenate(
                    [parent_child_data_dict[level][best_node],cluster_data],axis=0
            )
            # 对new parent node 进行 type name的预测
            representative_cluster_samples = extract_representative_samples(parent_child_rep_dict[level][new_parent_id],
                                                                            parent_child_data_dict[level][new_parent_id],
                                                    )
            parent_type_name, parent_type_description = generate_summary(representative_cluster_samples, type_desc,parent_type,parent_to_children)
            type_desc[parent_type_name] = parent_type_description
            type2id[parent_type_name] = new_parent_id
            id2type[new_parent_id] = parent_type_name


            #对他的两个子节点进行 type name的预测

            representative_cluster_samples = extract_representative_samples(parent_child_rep_dict[level][best_node],
                                                                            parent_child_data_dict[level][best_node],
                                                                            )
            type_name, description = generate_summary(representative_cluster_samples, type_desc,parent_type_name,parent_to_children)
            type_desc[type_name] = description
            type2id[type_name] = best_node
            id2type[best_node] = type_name


            representative_cluster_samples = extract_representative_samples(cluster_rep,
                                                                            cluster_data,
                                                                            )
            type_name, description = generate_summary(representative_cluster_samples, type_desc,parent_type_name,parent_to_children)
            type_desc[type_name] = description
            type2id[type_name] = cluster_id
            id2type[cluster_id] = type_name

            # ⭐ 标记成功
            merge_happened = True
            merge_node=best_node
        #     break
        else:
            # sim_max = max(sim,sim_max)
            path.append(best_node)
        # if found:
        #     break

    # if sim_max < 0.5:
    #     path=[999]
    return path, best_level,merge_happened,new_parent_id,merge_node


def update_tree_with_unknown(
    path,
    cluster_rep,
    cluster_data,
    cluster_id,
    parent_child_dict,
    parent_child_rep_dict,
    parent_child_data_dict,
    level_num,merge_happened, merge_node
):
    """把 unknown 插入 tree"""
    for i in range(level_num, level_num - len(path), -1):
        node = path[level_num - i]

        parent_child_rep_dict[i][node] = np.concatenate(
            [parent_child_rep_dict[i][node], cluster_rep],
            axis=0
        )
        parent_child_data_dict[i][node] = np.concatenate(
            [parent_child_data_dict[i][node], cluster_data],
            axis=0
        )

    if path[-1] not in parent_child_dict[level_num - len(path)+1].keys():
        parent_child_dict[level_num - len(path) + 1][path[-1]]=[cluster_id]
    else:
        parent_child_dict[level_num - len(path)+1][path[-1]].append(cluster_id)
    insert_level = level_num - len(path)
    if insert_level !=0 :
        parent_child_rep_dict[level_num - len(path)][cluster_id]=cluster_rep
        parent_child_data_dict[level_num - len(path)][cluster_id] = cluster_data
    # if merge_happened and merge_node is not None:
    #
    #     child_rep = parent_child_rep_dict[insert_level][reparent_node]
    #
    #     parent_child_rep_dict[insert_level][cluster_id] = np.concatenate(
    #         [parent_child_rep_dict[insert_level][cluster_id], child_rep],
    #         axis=0
    #     )

def convert_path_to_names(path, id2type,unknown_id_label):
    """id → type name"""
    result = []
    for i, node in enumerate(path):
        if i == 0:
            result.append('root')
        else:
            if node in id2type.keys():
                result.append(id2type[node])
            else:
                result.extend(unknown_id_label[node-100])
    return result


# ---------- 主函数 ----------
def get_parent_paths(label_path):
    parts = label_path.strip('/').split('/')
    paths = ['root']
    current = ''
    if label_path.count('/') ==1:
        return paths
    # 去掉最后一个节点
    for p in parts[:-1]:
        current += '/' + p
        paths.append(current)
    return paths
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def get_top_tf_keywords(cluster_samples, top_n=5):
    all_words = []
    for sample in cluster_samples:
        tokens = sample["sentence"]
        tokens = [token.lower() for token in tokens]
        tokens = [
            token for token in tokens
            if token not in stop_words
            and token.isalpha()
        ]
        all_words.extend(tokens)
    counter = Counter(all_words)
    keywords = [w for w, _ in counter.most_common(top_n)]
    return keywords

def ClusterMerge(
    clusterResult,
    all_idxes,
    all_feats,
    match_result,
    match_result_unlabeled,
    concept_type_path,
    ins_type_path,
    id2type,
    type2id,
    types,
    unknown_cluster_filter,
    unknown_id_label,
    type_desc
):

    cluster_ins_re_dict = clusterResult['cluster_ins_re_dict']
    cluster_ins_data_dict = clusterResult['cluster_ins_data_dict']
    cluster_labeled_feats_dict = clusterResult['cluster_labeled_feats_dict']
    level_num = len(concept_type_path[0])

    parent_to_children = defaultdict(list)

    for t in types:

        parts = t.strip("/").split("/")

        if len(parts) == 1:
            continue

        parent = "/" + "/".join(parts[:-1])

        parent_to_children[parent].append(t)

    print(parent_to_children)

    # 1️⃣ 构建 mapping
    cluster_map = build_cluster_map(match_result, concept_type_path, ins_type_path)
    cluster_map_gold = build_cluster_map(match_result_unlabeled, concept_type_path, ins_type_path)

    # 2️⃣ refine path
    refined_paths = refine_type_path(ins_type_path, cluster_map, level_num)
    gold_paths = refine_type_path(ins_type_path, cluster_map_gold, level_num)

    # 3️⃣ 构建 hierarchy
    parent_child_dict, parent_child_rep_dict,parent_child_data_dict = build_parent_child_structure(
        refined_paths, cluster_ins_re_dict, cluster_ins_data_dict, level_num
    )
    original_parent_child_dict = copy.deepcopy(parent_child_dict)
    original_parent_child_rep_dict = copy.deepcopy(parent_child_rep_dict)
    build_root_node(parent_child_dict, parent_child_rep_dict, parent_child_data_dict,level_num)
    # for i in range(level_num-1,0,-1):

    # 4️⃣ 提取 unknown clusters
    unknown_ids, unknown_reps,unknown_data = extract_unknown_clusters(
        refined_paths, cluster_ins_re_dict,cluster_ins_data_dict
    )
    unknown_data_filter={}
    for i,id in enumerate(unknown_ids):
        if id-100 in unknown_id_label.keys():
            unknown_data_filter[id] =unknown_data[i]


    # 5️⃣ hierarchical search + merge
    unknown_predict = {unknown_ids[i]:[] for i in range(len(unknown_ids))}

    sizes = [rep.shape[0] for rep in unknown_reps]
    sorted_idx = np.argsort(-np.array(sizes))

    for idx in sorted_idx:

        cluster_id = unknown_ids[idx]


        if (cluster_id-100) not in unknown_id_label.keys():
            continue
        unknown_sim = [[] for j in range(len(unknown_reps))]
        for j in range(len(unknown_sim)):
            if unknown_ids[j] - 100 in unknown_id_label.keys():
                for i in range(level_num, 0, -1):
                    for k, v in parent_child_rep_dict[i].items():
                        if i == level_num:
                            type = 'root'
                        else:
                            if k in id2type.keys():
                                type = (id2type[k])
                            else:
                                type = unknown_id_label[k - 100][0]
                        unknown_sim[j].append(
                            (type, pairwise_cosine_similarity_score(unknown_reps[j], parent_child_rep_dict[i][k])))
        print(unknown_id_label[cluster_id - 100])
        if cluster_id==119:
            print('')
        path, best_level,merge_happened,new_parent_node,merge_node = hierarchical_search(
            unknown_reps[idx],
            unknown_data[idx],
            parent_child_dict,
            parent_child_rep_dict,
            parent_child_data_dict,
            original_parent_child_rep_dict,
            level_num,
            id2type,
            type2id,
            unknown_id_label,
            cluster_id,
            type_desc,
            parent_to_children,
            top_k=5,

        )


        print(unknown_id_label[cluster_id - 100][0])
        print(unknown_sim[idx])
        print('cluster id: '+str(cluster_id))

        update_tree_with_unknown(
            path,
            unknown_reps[idx],
            unknown_data[idx],
            cluster_id,
            parent_child_dict,
            parent_child_rep_dict,
            parent_child_data_dict,
            level_num,
            merge_happened, merge_node
        )



        unknown_predict[cluster_id] = convert_path_to_names(path, id2type, unknown_id_label)
        if merge_happened:
            current_type=unknown_id_label[cluster_id-100][0]
            parent_type = "/".join(current_type.split("/")[:-1])
            unknown_predict[cluster_id].append(parent_type)
            unknown_predict[merge_node].append(parent_type)

        else:
            parent_type = unknown_predict[cluster_id][-1]
            representative_cluster_samples = extract_representative_samples(unknown_reps[idx],
                                                                            unknown_data[idx])
            type_name, description = generate_summary(representative_cluster_samples, type_desc,parent_type,parent_to_children)
            type_desc[type_name] = description
            type2id[type_name] = cluster_id
            id2type[cluster_id] = type_name
            print('新添加：')
            print('id:'+str(cluster_id))
            print('type:'+type_name)
        print(unknown_predict[cluster_id])
        print()
        print()



    # 6️⃣ evaluation
    taxo_p_total = 0
    taxo_r_total = 0
    taxo_f1_total = 0
    count = 0
    print('predict')
    for i, cluster_id in enumerate(unknown_ids):
        if (cluster_id-100) in unknown_id_label.keys():
            pred = unknown_predict[cluster_id]
            pred=[t.lower() for t in pred]
            label = unknown_id_label[cluster_id-100][0].lower()
            ### 这样弄的话，我最后对于 type name的评测该怎么办呢
            if label.endswith("/padding"):
                label = label[:-len("/padding")]
            label = get_parent_paths(label)

            # ⭐ 防止空路径
            # if len(pred) == 0:
            #     pred = ['root']

            # === ancestor closure ===
            u_tp = set(pred)
            u_tg = set(label)

            inter = u_tp.intersection(u_tg)

            # === Taxo-P ===
            p = len(inter) / max(len(u_tp), 1)

            # === Taxo-R ===
            r = len(inter) / max(len(u_tg), 1)

            # === Taxo-F1 ===
            if p + r == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)

            taxo_p_total += p
            taxo_r_total += r
            taxo_f1_total += f1

            print(unknown_id_label[cluster_id-100][0])
            print("label:", label)
            print("pred :", pred)
            print(f"P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
            print("...........................")

            count += 1

        # === macro average ===
    taxo_p = taxo_p_total / max(count, 1)
    taxo_r = taxo_r_total / max(count, 1)
    taxo_f1 = taxo_f1_total / max(count, 1)

    print("Taxo-P:", taxo_p)
    print("Taxo-R:", taxo_r)
    print("Taxo-F1:", taxo_f1)
    print("count:", count)

    for i, cluster_id in enumerate(unknown_ids):
        if (cluster_id - 100) in unknown_id_label.keys():
            # Ground-truth label
            gt_label = unknown_id_label[cluster_id - 100][0]
            if 'padding' in gt_label:
                gt_label = gt_label.rsplit('/', 1)[0]
            # Generated type name
            pred_label = id2type[cluster_id]

            keywords = get_top_tf_keywords(unknown_data[i])

            # Optional normalization
            gt_label = gt_label.strip().lower()
            pred_label = pred_label.strip().lower()

            print(f"[GT ] {gt_label}")
            print(f"[PRED] {pred_label}")
            print(f"[KEY] {keywords}")
            print()



