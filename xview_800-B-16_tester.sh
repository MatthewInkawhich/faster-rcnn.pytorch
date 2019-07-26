#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u test_rpn_ff.py --dataset xview_800 --net res101 --cfg ./cfgs/xview/800_B_16.yml --checksession 1 --checkepoch 6 --checkpoint 27128 --cuda
	 
