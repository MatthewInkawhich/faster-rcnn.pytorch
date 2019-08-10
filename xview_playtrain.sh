#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u trainval_net.py \
			--dataset xview_800 \
			--net res101_custom2 \
			--cfg_file ./cfgs/xview/800_B_16.yml \
            --layer_cfg [3,4,8] \
			--s 99 \
			--disp_interval 1 \
			--bs 2 \
			--nw 0 \
			--lr 1e-3 \
			--lr_decay_step 8 \
			--epochs 10 \
			--cuda \
	 
