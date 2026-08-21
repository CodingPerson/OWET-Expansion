
python pretrain.py --dataset 'BBN' --device 1 --contra_type 'sup' --combine_type 'sup' --split_val 1 --train_type 'pretrain' --type_box_dim 128 \
                    --enable_type_inter 1 --type_inter_type 'taxo' --type_n_neg 25 --type_IoU_margin 0.5 --enable_type_desc 1 --type_taxo_extra 0.1 --type_taxo_alpha 1.0 \
                    --mini_type_size 0.5 --gen_box_method 'mlp' --lr_box 1e-4 --num_epoch 200 --type_taxo_margin 0.01 --type_taxo_eps -0.01
#nohup python -u train_DKCC_main_reverb.py > reverb0.8-retrain-s2.log 2>&1 &
