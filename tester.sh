#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python -u test_net.py \
			--dataset pascal_voc \
			--net res101 \
                        --checksession 1 \
                        --checkepoch 10 \
                        --checkpoint 625 \
			--cuda \
                        --mGPUs \
                        --vis
	 
