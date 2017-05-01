#!/bin/bash
python3 -B train.py --input $1 --model strong --epoch 300 --dataGen True
