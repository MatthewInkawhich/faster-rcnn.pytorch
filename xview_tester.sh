#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u test_net.py --dataset xview_700 --net res101 --cfg ./cfgs/xview/A.yml --checksession 1 --checkepoch 6 --checkpoint 12653 --cuda
	 
