#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u test_rpn_ff.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_A_16.yml --checksession 1 --checkepoch 6 --checkpoint 23376 --cuda
	 