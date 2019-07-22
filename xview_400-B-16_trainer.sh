#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python -u trainval_net.py \
			--dataset xview_400 \
			--net res101 \
			--cfg_file ./cfgs/xview/400_B_16.yml \
			--s 1 \
			--disp_interval 10 \
            --save_iter 2000 \
			--bs 32 \
			--nw 4 \
			--lr 1e-3 \
			--lr_decay_step 3 \
			--epochs 6 \
			--cuda \
            --mGPUs
	 
