#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python -u trainval_net.py \
			--dataset pascal_voc \
			--net res101 \
			--disp_interval 100 \
			--bs 16 \
			--nw 4 \
			--lr 1e-2 \
			--lr_decay_step 8 \
			--epochs 10 \
			--cuda \
                        --mGPUs \
                        --s 2 \
	 
