#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -u test_net_ff.py --dataset xview_600 --net res101_custom --cfg ./cfgs/xview/600_B_16.yml --checksession 3412 --checkepoch 6 --checkpoint 23376 --cuda
