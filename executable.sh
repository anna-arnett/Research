#!/bin/bash

#$ -M palarcon@nd.edu
#$ -m abe  
#$ -q gpu
#$ -l gpu=1

echo "This job is running on host: $HOSTNAME"

echo "This job was assigned GPU: $CUDA_VISIBLE_DEVICES"

python3 ./bts.py
