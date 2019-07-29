#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u trainval_net.py \
			--dataset xview_600 \
			--net res101 \
			--cfg_file ./cfgs/xview/600_A_8.yml \
			--s 1 \
			--disp_interval 10 \
            --save_iter 2000 \
			--bs 4 \
			--nw 4 \
			--lr 1e-3 \
			--lr_decay_step 3 \
			--epochs 6 \
			--cuda \
            --mGPUs
	 
