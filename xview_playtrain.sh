#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u trainval_net.py \
			--dataset xview_700 \
			--net res101 \
			--cfg_file ./cfgs/xview/A.yml \
			--s 99 \
			--disp_interval 1 \
			--bs 4 \
			--nw 0 \
			--lr 4e-3 \
			--lr_decay_step 8 \
			--epochs 10 \
			--cuda \
	 
