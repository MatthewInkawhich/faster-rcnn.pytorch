#!/bin/bash

echo "EPOCH:1"
CUDA_VISIBLE_DEVICES=3 python -u test_net.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_16.yml --checksession 1 --checkepoch 1 --checkpoint 23376 --cuda
echo "EPOCH:2"
CUDA_VISIBLE_DEVICES=3 python -u test_net.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_16.yml --checksession 1 --checkepoch 2 --checkpoint 23376 --cuda
echo "EPOCH:3"
CUDA_VISIBLE_DEVICES=3 python -u test_net.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_16.yml --checksession 1 --checkepoch 3 --checkpoint 23376 --cuda
echo "EPOCH:4"
CUDA_VISIBLE_DEVICES=3 python -u test_net.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_16.yml --checksession 1 --checkepoch 4 --checkpoint 23376 --cuda
echo "EPOCH:5"
CUDA_VISIBLE_DEVICES=3 python -u test_net.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_16.yml --checksession 1 --checkepoch 5 --checkpoint 23376 --cuda
echo "EPOCH:6"
CUDA_VISIBLE_DEVICES=3 python -u test_net.py --dataset xview_600 --net res101 --cfg ./cfgs/xview/600_B_16.yml --checksession 1 --checkepoch 6 --checkpoint 23376 --cuda
	 
