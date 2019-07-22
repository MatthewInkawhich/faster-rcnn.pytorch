#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -u test_net.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_24.yml --checksession 1 --checkepoch 6 --checkpoint 23376 --cuda
	 
