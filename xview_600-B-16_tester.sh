#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -u test_rpn_ff.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_16.yml --checksession 1 --checkepoch 6 --checkpoint 23376 --cuda
	 