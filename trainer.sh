#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python trainval_net.py \
			--dataset pascal_voc \
			--net res101 \
			--disp_interval 1 \
			--bs 4 \
			--nw 0 \
			--lr 4e-3 \
			--lr_decay_step 8 \
			--epochs 10 \
			--cuda 
