#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -u test_net_ff.py --dataset xview_200 --net res101 --cfg ./cfgs/xview/200_B_16.yml --checksession 1 --checkepoch 6 --checkpoint 18000 --cuda --vis
	 
