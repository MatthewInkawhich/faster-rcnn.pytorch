#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -u trainval_net.py \
			--dataset xview_200 \
			--net res101 \
			--cfg_file ./cfgs/xview/200_B_16.yml \
			--s 1 \
			--disp_interval 10 \
            --save_iter 2000 \
			--bs 64 \
			--nw 4 \
			--lr 1e-3 \
			--lr_decay_step 3 \
			--epochs 6 \
			--cuda \
            --mGPUs
	 
