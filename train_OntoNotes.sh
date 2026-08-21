
  python train.py --dataset 'OntoNotes' --split_val 1 --train_type 'train' --cluster_rep 'box' --cluster2box 'R2B2' \
            --enable_weight 0 --combine_type 'sup' --device 0 --num_epoch 100 \
            --pretrain_path 'log/OntoNotes_gpt5-4-margin0.02-eps-0.01/checkpoint_OntoNotes_lr_2e-05_latest.pth' \
            --single_box_model 0 --margin 10 --lr_box 1e-4 --alpha 1 --mini_ins_size 0.1 --mini_type_size 0.5 --dist_type 'BoxE' --box_self_adv 1 \
            --box_loss_type 'neg_sample' --adv_temp 16.0 --enable_box_inter 0 --enable_type_inter 1 --enable_cross_inter 1 --box_inter_type 'taxo' --inter_n_neg 0 --type_n_neg 25 --cross_n_neg 25 --proj_type2ins 0 \
            --type_taxo_extra 0.1 --type_taxo_alpha 1.0 --cross_taxo_extra 0.1 --cross_taxo_alpha 1.0 --box_inter_start -1 --cross_inter_start -1 --type_taxo_eps -0.01 --type_taxo_margin 0.02 --cross_taxo_eps -0.01 --cross_taxo_margin 0 --type_inter_type 'taxo' --cross_inter_type 'taxo' --gen_box_method 'mlp' --type_box_dim 128 \
            --proj_ins2type 1 --match_weight 0 --cross_IoU_margin 0.5 --onlyBottom 1 --use_type_emb 0 --init_type 1 --type_emb_method 'B4T' --cross_dynamic_weight 1 --type_IoU_margin 0.5 --enable_type_desc 1 --cross_weight 3 --enable_cross_pb 0 \
            --enable_active 1 --enable_oracle 0 --enable_refine 0 --random 0