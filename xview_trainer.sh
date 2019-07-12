#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python -u trainval_net.py \
			--dataset xview_700 \
			--net res101 \
			--cfg_file ./cfgs/xview/B.yml \
			--s 1 \
			--disp_interval 10 \
			--bs 8 \
			--nw 4 \
			--lr 1e-3 \
			--lr_decay_step 8 \
			--epochs 6 \
			--cuda \
            --mGPUs
	 
