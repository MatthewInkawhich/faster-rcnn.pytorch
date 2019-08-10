#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -u test_rpn_ff.py --dataset xview_600 --net res101_custom --cfg ./cfgs/xview/600_B_8.yml --checksession 31 --checkepoch 6 --checkpoint 46754 --cuda
#CUDA_VISIBLE_DEVICES=3 python -u test_rpn_ff.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_8.yml --checksession 1 --checkepoch 6 --checkpoint 8000 --cuda --vis
	 
