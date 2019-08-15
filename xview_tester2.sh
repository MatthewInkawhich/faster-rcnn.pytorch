#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -u test_net_ff.py --dataset xview_600 --net res101_custom2 --cfg ./cfgs/xview/600_A_16.yml --layer_cfg [3,4,23] --checksession 3423 --checkepoch 2 --checkpoint 23376 --cuda
