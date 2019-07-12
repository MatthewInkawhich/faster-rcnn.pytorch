#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u trainval_net.py \
			--dataset pascal_voc \
			--net res101 \
			--s 99 \
			--disp_interval 10 \
			--bs 4 \
			--nw 0 \
			--lr 4e-3 \
			--lr_decay_step 8 \
			--epochs 10 \
			--cuda \
	 
