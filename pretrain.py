import csv
import os

os.environ['PYTHONHASHSEED'] = '11'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

torch.use_deterministic_algorithms(True)
import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from torch.utils.data import DataLoader

import constant
from config import initConfig
from dataProcess.OWETDataset import labeled_collate_fn, OWETDataset
from dataProcess.loadData import loadDataset
from models.model import OWETModel
from utils.finch import FINCH, getFinchPred
from utils.hdbscanTools import hdbscanManager, MSTLinked, compute_mutual_reachability
from utils.metric import getB3Eval, getVmAndARIAndNMI
from utils.cluster_acc import test_agglo, cluster_acc, hdbscan_acc, finch_acc
from utils.forTrain import mask_tokens, get_optimizer
from utils.loss import *
from utils.losshistory import lossHistory
from utils.utils import setSeed, getType2Id, getLevelTarget, loadCheckConfig, saveCheckConfig, loadLossHistory, \
    saveLossHistory, getTypeNum, getTypesDesc, getTypesInput, getNegSampleList
from utils.view_generator import view_generator


def pad_tensors(tensor1, tensor2, padding_value=0):
    # 获取两个张量的最大长度
    max_len = max(tensor1.size(1), tensor2.size(1))

    # 对 tensor1 进行填充
    if tensor1.size(1) < max_len:
        padding = torch.zeros(tensor1.size(0), max_len - tensor1.size(1), dtype=tensor1.dtype, device=tensor1.device)
        tensor1 = torch.cat([tensor1, padding], dim=1)

    # 对 tensor2 进行填充
    if tensor2.size(1) < max_len:
        padding = torch.zeros(tensor2.size(0), max_len - tensor2.size(1), dtype=tensor2.dtype, device=tensor2.device)
        tensor2 = torch.cat([tensor2, padding], dim=1)

    return tensor1, tensor2


def compare_np_state(a, b):
    return (a[0] == b[0]
            and np.array_equal(a[1], b[1])
            and a[2] == b[2]
            and a[3] == b[3]
            and a[4] == b[4])


def judge_random_change(a, b, c, d):
    init_a_ = random.getstate()
    init_b_ = np.random.get_state()
    init_c_ = torch.get_rng_state()
    init_d_ = torch.cuda.get_rng_state()
    if init_a_ != a:
        return False, 0
    if not compare_np_state(b, init_b_):
        return False, 1
    if not torch.equal(c, init_c_):
        return False, 2
    if not torch.equal(d, init_d_):
        return False, 3
    return True, -1


init_a = None
init_b = None
init_c = None
init_d = None
info = ['random change', 'np random change', 'torch random change', 'torch cuda random change']


def transform_type_inputs(type_inputs):
    input_ids = [item['input_ids'].squeeze(0) for item in type_inputs]
    attention_mask = [item['attention_mask'].squeeze(0) for item in type_inputs]
    token_type_ids = [item['token_type_ids'].squeeze(0) for item in type_inputs]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

    return input_ids, attention_mask, token_type_ids


def train(model, device, optimizer, scheduler, labeled_dataloader, adapted_dataloader, lossM,
          eval_dataloader, eval_dataset, val_dataloader, id2type, losshistory, args, save_dir='log', types_input=None,
          neg_list=None):
    if args.start_epoch == 0:
        clusterAccDict, b3Dict, VmARINMIDict = toEval(model, device, eval_dataloader, eval_dataset, id2type, 0,
                                                      args, mode='all')
        losshistory.append_eval(clusterAccDict, b3Dict, VmARINMIDict)
    # global init_a
    # global init_b
    # global init_c
    # global init_d
    # global info
    # init_a = random.getstate()
    # init_b = np.random.get_state()
    # init_c = torch.get_rng_state()
    # init_d = torch.cuda.get_rng_state()
    # info = ['random change', 'np random change', 'torch random change', 'torch cuda random change']

    cur_epoch = args.start_epoch
    start = args.start_epoch

    type_inter = Type_box_intersection(margin=args.type_taxo_margin, eps=args.type_taxo_eps)
    type_batch = transform_type_inputs(types_input)
    type_batch = tuple(t.to(device) for t in type_batch)

    best_lab_acc = 0
    best_epoch = 0
    dir_path = os.path.join(save_dir, 'val')

    for epoch in range(start, args.num_epoch):
        # args.labeled_g.manual_seed(args.seed + epoch)
        # args.adapted_g.manual_seed(args.seed + epoch)
        # args.cpu_g.manual_seed(args.seed + epoch)
        # args.gpu_g.manual_seed(args.seed + epoch)

        labelediter = iter(labeled_dataloader)
        # adaptededitor = iter(adapted_dataloader)
        model.train()
        train_loss = 0
        sup_loss = 0
        mlm_loss = 0
        ce_loss = 0
        self_loss = 0
        type_loss = 0
        re_type_loss = 0
        train_steps = 0
        loss_re_type=0

        # judge, idx_aaa = judge_random_change(init_a, init_b, init_c, init_d)
        # if not judge:
        #     with open('random_change_log.txt', 'a', encoding='utf-8') as f:
        #         f.write(f'{args.dataset}--epoch:{epoch}--{info[idx_aaa]}')
        #         if idx_aaa == 2:
        #             init_c = torch.get_rng_state()
        #             f.write(f'--test_again_torch_random--')
        #             judge, idx_aaa = judge_random_change(init_a, init_b, init_c, init_d)
        #             f.write(f'{str(judge)}\n')
        #         else:
        #             f.write('\n')

        if args.contra_type == 'sup':
            bar = tqdm.tqdm(enumerate(adapted_dataloader), total=len(adapted_dataloader), desc='pretrain one epoch')
            for step, batch in bar:
                postDict = dict()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                mask_ids, mask_lb = mask_tokens(input_ids.cpu(), model.tokenizer, args.cpu_g,
                                                mlm_probability=args.mask_prob)
                mask_ids, mask_lb = mask_ids.to(device), mask_lb.to(device)
                loss_mlm = model(mask_ids, input_mask, segment_ids, labels=mask_lb, mode='mlm', generator=args.gpu_g)
                postDict['mlm loss'] = loss_mlm.item()
                # loss_mlm = torch.tensor(0)

                try:
                    batch = next(labelediter)
                except StopIteration:
                    # print(f'step {step} run iter')
                    labelediter = iter(labeled_dataloader)
                    batch = next(labelediter)

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                targets = label_ids.cpu().numpy()
                level_target = getLevelTarget(targets, args.data_level, id2type)
                lab_label_ids_ = torch.tensor(level_target[-1], dtype=torch.long).to(device)

                feats, logits = model(input_ids, segment_ids, input_mask, mode='train', generator=args.gpu_g)

                if args.combine_type in ['tot', 'ce']:
                    loss_ce = lossM.ce_loss(logits, lab_label_ids_)
                    postDict['ce loss'] = loss_ce.item()
                else:
                    loss_ce = torch.tensor(0)

                if args.combine_type in ['tot', 'sup']:
                    loss_sup = lossM.sup_loss(feats, device, label_ids)
                    postDict['sup loss'] = loss_sup.item()
                else:
                    loss_sup = torch.tensor(0)

                if args.enable_type_inter == 1:
                    input_ids, input_mask, segment_ids = type_batch
                    type_centers, type_offsets = model(input_ids, segment_ids, input_mask, mode='type',
                                                       generator=args.gpu_g)
                    pred_logits1, mini_logits1 = regularization_logits(type_offsets, args.mini_type_size)
                    loss_re_type = lossM.re_loss(pred_logits1, mini_logits1)
                    postDict['re type loss'] = loss_re_type.item()
                    #loss_re_type=torch.tensor(0)
                    result = type_inter(type_centers, type_offsets, args.type_n_neg, neg_list,
                                        args.gumbel_beta, args.tanh, args.type_taxo_alpha, args.type_taxo_extra,
                                        mode=args.type_inter_type, IoU_mode=args.type_IoU_mode,
                                        IoU_margin=args.type_IoU_margin, d_alpha=args.type_d_alpha)
                    if args.type_inter_type == 'B4T':
                        loss_type = 0.5 * lossM.type_loss(result[0], result[1])
                    elif args.type_inter_type == 'taxo':
                        loss_type = result[0]
                    elif args.type_inter_type == 'box+':
                        loss_type = result[0]
                    elif args.type_inter_type == 'IoU':
                        loss_type = result
                    elif args.type_inter_type == 'cbox':
                        loss_type = result
                    postDict['type loss'] = loss_type.item()
                else:
                    loss_type=torch.tensor(0)
                    loss_re_type=torch.tensor(0)

                loss = loss_sup + loss_mlm + loss_ce + loss_type + loss_re_type

                train_loss += loss.item()
                sup_loss += loss_sup.item()
                mlm_loss += loss_mlm.item()
                ce_loss += loss_ce.item()
                type_loss += loss_type.item()
                re_type_loss += loss_re_type.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_steps += 1
                bar.set_postfix(postDict)
        elif args.contra_type == 'self':
            bar = tqdm.tqdm(enumerate(adapted_dataloader), total=len(adapted_dataloader), desc='pretrain one epoch')
            for step, batch in bar:
                postDict = dict()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                # batch_size = unlab_label_ids.size(0)
                # unlab_label_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)
                try:
                    batch = next(labelediter)
                except StopIteration:
                    labelediter = iter(labeled_dataloader)
                    batch = next(labelediter)
                batch = tuple(t.to(device) for t in batch)
                lab_input_ids, lab_input_mask, lab_segment_ids, lab_label_ids, _ = batch
                targets = lab_label_ids.cpu().numpy()
                level_target = getLevelTarget(targets, args.data_level, id2type)
                lab_label_ids_ = torch.tensor(level_target[-1], dtype=torch.long).to(device)

                feats, logits = model(lab_input_ids, lab_segment_ids, lab_input_mask, mode='train')
                if args.combine_type in ['tot', 'ce']:
                    loss_ce = lossM.ce_loss(logits, lab_label_ids_)
                    postDict['ce loss'] = loss_ce.item()
                else:
                    loss_ce = torch.tensor(0)

                if args.combine_type in ['tot', 'sup']:
                    loss_sup = lossM.sup_loss(feats, device, lab_label_ids)
                    postDict['sup loss'] = loss_sup.item()
                else:
                    loss_sup = torch.tensor(0)

                if args.view_strategy == "rtr":
                    input_ids_a = input_ids
                    input_ids_b = args.view_generator.random_token_replace(input_ids.cpu()).to(device)
                elif args.view_strategy == "none":
                    input_ids_a = input_ids
                    input_ids_b = input_ids
                else:
                    raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")

                # input_ids = torch.cat((lab_input_ids, unlab_input_ids))
                # input_mask = torch.cat((lab_input_mask, unlab_input_mask))
                # segment_ids = torch.cat((lab_segment_ids, unlab_segment_ids))
                # label_ids = torch.cat((lab_label_ids_, unlab_label_ids))

                aug_a, _ = model(input_ids_a, segment_ids, input_mask, mode='train')
                aug_b, _ = model(input_ids_b, segment_ids, input_mask, mode='train')
                batch_size = aug_a.shape[0]

                norm_logits = torch.nn.functional.normalize(aug_a)
                # norm_logits=aug_a
                norm_aug_logits = torch.nn.functional.normalize(aug_b)
                # norm_aug_logits=aug_b
                # labels_expand = label_ids.expand(batch_size, batch_size)
                # mask = torch.eq(labels_expand, labels_expand.T).long()
                # mask[label_ids == -1, :] = 0
                #
                # logits_mask = torch.scatter(mask, 0, torch.arange(batch_size).unsqueeze(0).to(device), 1)
                logits_mask = torch.eye(batch_size, dtype=torch.long).to(device)

                contrastive_logits = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim=1)
                loss_self = lossM.self_loss(contrastive_logits, mask=logits_mask, device=device)
                # loss_self = lossM.self_loss(norm_logits,norm_aug_logits,logits_mask)
                postDict['self loss'] = loss_self.item()

                loss = loss_ce + loss_sup + loss_self

                train_loss += loss.item()
                sup_loss += loss_sup.item()
                ce_loss += loss_ce.item()
                self_loss += loss_self.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_steps += 1
                bar.set_postfix(postDict)
        elif args.contra_type == 'test':
            bar = tqdm.tqdm(enumerate(adapted_dataloader), total=len(adapted_dataloader), desc='pretrain one epoch')
            for step, batch in bar:
                postDict = dict()
                if args.combine_type == 'mlm':
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, _ = batch
                    mask_ids, mask_lb = mask_tokens(input_ids.cpu(), model.tokenizer, args.cpu_g,
                                                    mlm_probability=args.mask_prob)
                    mask_ids, mask_lb = mask_ids.to(device), mask_lb.to(device)
                    loss_mlm = model(mask_ids, input_mask, segment_ids, labels=mask_lb, mode='mlm',
                                     generator=args.gpu_g)
                    postDict['mlm loss'] = loss_mlm.item()
                else:
                    loss_mlm = torch.tensor(0)

                if args.combine_type in ['tot', 'ce', 'sup']:
                    try:
                        batch = next(labelediter)
                    except StopIteration:
                        # print(f'step {step} run iter')
                        labelediter = iter(labeled_dataloader)
                        batch = next(labelediter)

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, _ = batch

                    targets = label_ids.cpu().numpy()
                    level_target = getLevelTarget(targets, args.data_level, id2type)
                    lab_label_ids_ = torch.tensor(level_target[-1], dtype=torch.long).to(device)

                    feats, logits = model(input_ids, segment_ids, input_mask, mode='train', generator=args.gpu_g)

                    if args.combine_type in ['tot', 'ce']:
                        loss_ce = lossM.ce_loss(logits, lab_label_ids_)
                        postDict['ce loss'] = loss_ce.item()
                    else:
                        loss_ce = torch.tensor(0)

                    if args.combine_type in ['tot', 'sup']:
                        loss_sup = lossM.sup_loss(feats, device, label_ids)
                        postDict['sup loss'] = loss_sup.item()
                    else:
                        loss_sup = torch.tensor(0)
                else:
                    loss_ce = torch.tensor(0)
                    loss_sup = torch.tensor(0)

                loss = loss_sup + loss_mlm + loss_ce

                train_loss += loss.item()
                sup_loss += loss_sup.item()
                mlm_loss += loss_mlm.item()
                ce_loss += loss_ce.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_steps += 1
                bar.set_postfix(postDict)
        else:
            raise Exception(f'not contra type {args.contra_type}')

        loss = train_loss / train_steps
        sup_loss = sup_loss / train_steps
        mlm_loss = mlm_loss / train_steps
        ce_loss = ce_loss / train_steps
        self_loss = self_loss / train_steps
        type_loss = type_loss / train_steps
        re_type_loss = re_type_loss / train_steps
        if args.contra_type == 'sup':
            print(
                f'Epoch {epoch} train_loss: {loss} sup loss: {sup_loss} mlm loss: {mlm_loss} type loss: {type_loss} re type loss: {re_type_loss}')
        elif args.contra_type == 'self':
            print(f'Epoch {epoch} train_loss: {loss} sup loss: {sup_loss} ce loss: {ce_loss} self loss: {self_loss}')

        cur_epoch += 1

        if args.split_val:
            pass
            # lab_acc = 0
            # for cluster_type in ['HAC', 'HDBSCAN']:
            #     args.cluster_type = cluster_type
            #     lab_acc_, b3Dict, VmARINMIDict = toVal(model, device, val_dataloader, id2type, epoch, args)
            #     losshistory.append_val(lab_acc_, b3Dict, VmARINMIDict)
            #     lab_acc = max(lab_acc_, lab_acc)
            # lab_acc, b3Dict, VmARINMIDict = toVal(model, device, val_dataloader, id2type, epoch, args)
            # losshistory.append_val(lab_acc, b3Dict, VmARINMIDict)
            # if lab_acc > best_lab_acc:
            #     best_epoch = cur_epoch
            #     with open(f'{dir_path}/best.txt', 'w', encoding='utf-8') as f:
            #         f.write(f'best epoch is {best_epoch}\n')
            #         f.write(
            #             f'lab acc: {lab_acc};b3 prec: {b3Dict[0][0]}; b3 recall: {b3Dict[0][1]}; b3 f1: {b3Dict[0][2]}'
            #             f'v_measure: {VmARINMIDict[0][0]}; homogeneity: {VmARINMIDict[0][1]}; '
            #             f'completeness: {VmARINMIDict[0][2]}; ari: {VmARINMIDict[0][3]}; nmi: {VmARINMIDict[0][4]}\n')
            #     best_lab_acc = lab_acc
            #     if args.contra_type != 'test':
            #         checkpoint = {
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'scheduler_state_dict': scheduler.state_dict(),
            #         }
            #         torch.save(checkpoint, f'{save_dir}/checkpoint_{args.dataset}_lr_{args.lr}_valBestAcc.pth')
        # 0 lab 1 unlab 2 unlab_known 3 unlab_unknown
        if cur_epoch % 10 == 0:
            # for cluster_type in ['HAC', 'HDBSCAN']:
            #     args.cluster_type = cluster_type
            #     clusterAccDict, b3Dict, VmARINMIDict = toEval(model, device, eval_dataloader,
            #                                                   eval_dataset, id2type, epoch, args,
            #                                                   mode='all')
            #     if args.contra_type == 'sup':
            #         losshistory.append_eval(clusterAccDict, b3Dict, VmARINMIDict)
            #     elif args.contra_type in ['self', 'test']:
            #         losshistory.append_eval(clusterAccDict, b3Dict, VmARINMIDict)
            clusterAccDict, b3Dict, VmARINMIDict = toEval(model, device, eval_dataloader,
                                                          eval_dataset, id2type, epoch, args,
                                                          mode='all')
            losshistory.append_eval(clusterAccDict, b3Dict, VmARINMIDict)

        if args.contra_type == 'sup':
            losshistory.append_loss(loss, sup_loss, mlm_loss, ce_loss)
        elif args.contra_type in ['self', 'test']:
            losshistory.append_loss(loss, sup_loss, ce_loss, self_loss)

        if cur_epoch % 10 == 0:
            # if args.contra_type != 'test':
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, f'{save_dir}/checkpoint_{args.dataset}_lr_{args.lr}_latest.pth')
            saveLossHistory(lossHistory, f'{save_dir}/lossHistory.pkl')
            args.start_epoch = cur_epoch
            saveCheckConfig(args, f'{save_dir}/config.pkl')
            losshistory.loss_plot()
            losshistory.record()
        if cur_epoch == args.stop_epoch:
            return


def toVal(model, device, dataloader, id2type, epoch, args):
    with torch.no_grad():
        model.eval()
        all_feats = []
        targets = []

        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                        desc='val one epoch of extra feature')
        for step, batch in bar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, idxes = batch
            feats, _ = model(input_ids, segment_ids, input_mask, label_ids, mode="train")

            all_feats.extend(feats.detach().cpu().numpy())
            targets.extend(label_ids.cpu().numpy())

        targets = np.asarray(targets)
        all_feats = np.asarray(all_feats)
        level_target = getLevelTarget(targets, args.data_level, id2type)
        class_num = len(np.unique(level_target[args.data_level - 1]))

        # cluster acc
        print(args.cluster_type)
        if args.cluster_type == 'HAC':
            linked = linkage(all_feats, method=args.cluster_method)
        elif args.cluster_type == 'HDBSCAN':
            dist_martix = compute_mutual_reachability(all_feats, min_samples=5)
            linked = linkage(dist_martix, method=args.cluster_method)
        # elif args.cluster_type == 'FINCH':
        #     c, num_clust = FINCH(all_feats)
        else:
            raise Exception(f'args.cluster_type unsupported!')

        if args.cluster_type in ['HAC', 'HDBSCAN']:
            d = linked[:, 2][-class_num]
            preds = fcluster(linked, t=d, criterion='distance')
        # elif args.cluster_type == 'FINCH':
        #     preds = getFinchPred(c, num_clust, all_feats, args.num_val_class)
        else:
            raise Exception(f'args.cluster_type unsupported!')

        lab_acc = cluster_acc(y_true=level_target[args.data_level - 1], y_pred=preds)
        print('val==>:')
        print(f'epoch {epoch}: lab acc:{lab_acc}')

        targetList = [level_target[args.data_level - 1]]
        predList = [preds]

        # B3
        b3Dict = getB3Eval(targetList, predList)
        print(f'epoch {epoch}: b3 prec: {b3Dict[0][0]}; b3 recall: {b3Dict[0][1]}; '
              f'b3 f1: {b3Dict[0][2]}')

        # V_measure and ARI
        VmARINMIDict = getVmAndARIAndNMI(targetList, predList)
        print(f'epoch {epoch}: v_measure: {VmARINMIDict[0][0]}; homogeneity: {VmARINMIDict[0][1]}; '
              f'completeness: {VmARINMIDict[0][2]}; ari: {VmARINMIDict[0][3]}; nmi: {VmARINMIDict[0][4]}')

        return lab_acc, b3Dict, VmARINMIDict


def toEval(model, device, dataloader, dataset, id2type, epoch, args, mode='one', unlab_split=True):
    with torch.no_grad():
        model.eval()
        all_feats = []
        targets = []
        mask_lab = np.array([])
        mask_unlab_known = np.array([])
        mask_extra_known_and_label = np.array([])
        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                        desc='eval one epoch of extra feature')
        for step, batch in bar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, idxes = batch
            feats, _ = model(input_ids, segment_ids, input_mask, label_ids, mode="train")

            all_feats.extend(feats.detach().cpu().numpy())
            targets.extend(label_ids.cpu().numpy())
            mask_lab = np.append(mask_lab, np.array(
                [True if dataset.data[i]['mention']['dtype'] == 'known_labeled' else False
                 for i in idxes]))
            mask_unlab_known = np.append(mask_unlab_known, np.array(
                [True if dataset.data[i]['mention']['dtype'] == 'known_unlabeled' else False
                 for i in idxes]))
            mask_extra_known_and_label = np.append(mask_extra_known_and_label, np.array(
                [True if dataloader.dataset.data[i]['mention']['dtype'] == 'extra' or
                         dataloader.dataset.data[i]['mention']['dtype'] == 'known_labeled' else False
                 for i in idxes]))

        targets = np.asarray(targets)
        all_feats = np.asarray(all_feats)
        level_target = getLevelTarget(targets, args.data_level, id2type)
        mask_lab = mask_lab.astype(bool)
        mask_unlab_known = mask_unlab_known.astype(bool)
        mask_extra_known_and_label = mask_extra_known_and_label.astype(bool)


        # cluster acc
        clusterAccDict = dict()
        print(args.cluster_type)
        if args.cluster_type == 'HAC':
            linked = linkage(all_feats, method=args.cluster_method)
        elif args.cluster_type == 'HDBSCAN':
            dist_martix = compute_mutual_reachability(all_feats, min_samples=5)
            linked = linkage(dist_martix, method=args.cluster_method)
        elif args.cluster_type == 'FINCH':
            c, num_clust = FINCH(all_feats)
        else:
            raise Exception(f'args.cluster_type error {args.cluster_type}')

        if args.cluster_type in ['HAC', 'HDBSCAN']:
            level_acc, best_level_k, preds = test_agglo(linked, level_target, mask_lab, mask_unlab_known, mask_extra_known_and_label,args,
                                                        rePred=True)
        elif args.cluster_type == 'FINCH':
            level_acc, best_level_k, preds = finch_acc(c, num_clust, all_feats, level_target, mask_lab,
                                                       mask_unlab_known, args,
                                                       rePred=True)
        else:
            raise Exception(f'args.cluster_type error {args.cluster_type}')

        lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc = level_acc[args.data_level - 1]
        best_k = best_level_k[args.data_level - 1]
        clusterAccDict[0] = [tot_acc, unlab_known_acc, unlab_unknown_acc, lab_acc, best_k]
        print('eval==>:')
        print(f'epoch {epoch}: tot_acc: {tot_acc}; unlab_known_acc: {unlab_known_acc}; '
              f'unlab_unknown_acc: {unlab_unknown_acc}; lab_acc: {lab_acc}; '
              f'best_k: {best_k}')

        if mode == 'all':
            for base in range(1, 4):
                if args.cluster_type in ['HAC', 'HDBSCAN']:
                    level_acc, best_level_k, _ = test_agglo(linked, level_target, mask_lab, mask_unlab_known, mask_extra_known_and_label,args,
                                                            base=base)
                elif args.cluster_type == 'FINCH':
                    level_acc, best_level_k, _ = finch_acc(c, num_clust, all_feats, level_target, mask_lab,
                                                           mask_unlab_known, args,
                                                           base=base)
                else:
                    raise Exception(f'args.cluster_type error {args.cluster_type}')

                lab, tot, unlab_known, unlab_unknown = level_acc[args.data_level - 1]
                k = best_level_k[args.data_level - 1]
                clusterAccDict[base] = [tot, unlab_known, unlab_unknown, lab, k]
        elif mode == 'one':
            pass
        else:
            raise Exception(f'toEval mode param unsupported!')

        target = level_target[args.data_level - 1]

        # labeled
        target_lab = target[mask_lab]
        pred_lab = preds[mask_lab]
        # unlabeled
        target_unlab = target[~mask_extra_known_and_label]
        pred_unlab = preds[~mask_extra_known_and_label]
        mask_unlab_known = mask_unlab_known[~mask_extra_known_and_label]

        targetList = [target_lab, target_unlab]
        predList = [pred_lab, pred_unlab]
        target_unlab_known = target_unlab[mask_unlab_known]
        pred_unlab_known = pred_unlab[mask_unlab_known]
        target_unlab_unknown = target_unlab[~mask_unlab_known]
        pred_unlab_unknown = pred_unlab[~mask_unlab_known]
        targetList.extend([target_unlab_known, target_unlab_unknown])
        predList.extend(([pred_unlab_known, pred_unlab_unknown]))

        # B3
        b3Dict = getB3Eval(targetList, predList)
        print(f'epoch {epoch}: b3 prec: {b3Dict[0][0]}; b3 recall: {b3Dict[0][1]}; '
              f'b3 f1: {b3Dict[0][2]}')

        # V_measure and ARI
        VmARINMIDict = getVmAndARIAndNMI(targetList, predList)
        print(f'epoch {epoch}: v_measure: {VmARINMIDict[0][0]}; homogeneity: {VmARINMIDict[0][1]}; '
              f'completeness: {VmARINMIDict[0][2]}; ari: {VmARINMIDict[0][3]}; nmi: {VmARINMIDict[0][4]}')

        return clusterAccDict, b3Dict, VmARINMIDict


def main():
    args = initConfig()
    setSeed(args.seed, args.n_gpu)
    args.cpu_g = torch.Generator(device='cpu').manual_seed(args.seed)
    args.gpu_g = torch.Generator(device=f'cuda:{args.device}').manual_seed(args.seed)
    args.gold_k_list = constant.dataset_gold_k[args.dataset]

    save_dir = 'log'
    args.start_epoch = 0

    # 加载checkpoint
    checkpointDir = ''
    if checkpointDir != '':
        checkpointPath = None
        checkpointConfigPath = None
        lossHistoryPath = None
        for filename in os.listdir(checkpointDir):
            if filename.endswith('latest.pth'):
                checkpointPath = os.path.join(checkpointDir, filename)
                break
        for filename in os.listdir(checkpointDir):
            if filename.endswith('config.pkl'):
                checkpointConfigPath = os.path.join(checkpointDir, filename)
                break
        for filename in os.listdir(checkpointDir):
            if filename.endswith('lossHistory.pkl'):
                checkpointConfigPath = os.path.join(checkpointDir, filename)
                break
        if checkpointPath is None or checkpointConfigPath is None or lossHistoryPath is None:
            raise Exception('checkpoint not find or info  incomplete!')
        checkpoint = torch.load(checkpointPath)
        saved_args = loadCheckConfig(checkpointConfigPath)
        for key, value in vars(saved_args).items():
            if hasattr(args, key):
                setattr(args, key, value)
        losshistory = loadLossHistory(lossHistoryPath)
        save_dir = checkpointDir

    # 初始化一些参数
    args.num_known_class = constant.type_nums[args.dataset]['pad']['known']
    args.num_unknown_class = constant.type_nums[args.dataset]['pad']['unknown']
    args.data_level = constant.level_num[args.dataset]

    type2id = getType2Id(constant.type2id_path[args.dataset][1])
    id2type = {v: k for k, v in type2id.items()}
    model = OWETModel(args)
    print(args.num_known_class)
    print(type2id)
    # 加载数据
    train_data, labeled_data, val_data,extra_known = loadDataset(args.dataset, args.seed, args.split_val, args.val_ratio,args.active_budget)
    print('labeled_data')
    print(len(labeled_data))
    print('val_data')
    print(len(val_data))
    labeled_dataset = OWETDataset(labeled_data, type2id, model.tokenizer, num_classes=args.num_known_class, args=args,
                                  mode='labeled')
    labeled_g = torch.Generator().manual_seed(args.seed)
    args.labeled_g = labeled_g
    sampler = torch.utils.data.WeightedRandomSampler(labeled_dataset.sample_weights, num_samples=len(labeled_dataset),
                                                     generator=labeled_g)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.labeled_batch_size, shuffle=False, sampler=sampler,
                                    drop_last=True, collate_fn=labeled_collate_fn, num_workers=0, generator=labeled_g)
    if args.split_val:
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

    # 有监督和自监督的dataloader
    if args.contra_type == 'sup':
        adapted_data = train_data + labeled_data+extra_known
        adapted_dataset = OWETDataset(adapted_data, type2id, model.tokenizer, num_classes=args.num_known_class,
                                      args=args,
                                      mode='unlabeled')
        adapted_g = torch.Generator().manual_seed(args.seed)
        args.adapted_g = adapted_g
        sampler = torch.utils.data.RandomSampler(adapted_dataset, num_samples=len(adapted_dataset), generator=adapted_g)
        adapted_dataloader = DataLoader(adapted_dataset, batch_size=args.mlm_batch_size, shuffle=False,
                                        drop_last=True, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                        generator=adapted_g)
    elif args.contra_type in ['self', 'test']:
        adapted_data = train_data + labeled_data+extra_known
        adapted_dataset = OWETDataset(adapted_data, type2id, model.tokenizer, num_classes=args.num_known_class,
                                      args=args,
                                      mode='unlabeled')
        adapted_g = torch.Generator().manual_seed(args.seed)
        args.adapted_g = adapted_g
        sampler = torch.utils.data.RandomSampler(adapted_dataset, num_samples=len(adapted_dataset), generator=adapted_g)
        adapted_dataloader = DataLoader(adapted_dataset, batch_size=args.self_batch_size, shuffle=False,
                                        drop_last=True, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                        generator=adapted_g)
    else:
        raise Exception(f'no contra type {args.contra_type}')

    # eval的dataloader
    total_data = train_data + labeled_data+extra_known
    eval_dataset = OWETDataset(total_data, type2id, model.tokenizer,
                               num_classes=args.num_known_class + args.num_unknown_class, args=args,
                               mode='labeled')
    sampler = torch.utils.data.SequentialSampler(eval_dataset)
    eval_g = torch.Generator().manual_seed(args.seed)
    args.eval_g = eval_g
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 drop_last=False, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                 generator=eval_g)

    # 获取相关
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    args.epoch_steps = len(adapted_dataloader)
    optimizer, scheduler = get_optimizer(model, args, mode='bert')
    lossM = lossManager(args)

    # 载入checkpoint
    if checkpointDir != '':
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'log/{args.dataset}_{current_time}'
        losshistory = lossHistory(save_dir, args.contra_type, args.train_type)
        losshistory.gold_k = constant.dataset_gold_k[args.dataset][0]

    for key, value in vars(args).items():
        print(f"{key}:{value}")

    args.view_generator = view_generator(model.tokenizer, args)

    if args.enable_type_desc == 1:
        type_desc = getTypesDesc(constant.type_description[args.dataset])
    else:
        type_desc = None

    types, types_input = getTypesInput(constant.type2id_path[args.dataset][1], args.num_known_class, model.tokenizer,
                                       type_desc)
    neg_list = getNegSampleList(types, type2id)

    train(model, device, optimizer, scheduler, labeled_dataloader, adapted_dataloader, lossM,
          eval_dataloader, eval_dataset, val_dataloader, id2type, losshistory, args, save_dir, types_input, neg_list)


if __name__ == '__main__':
    main()
