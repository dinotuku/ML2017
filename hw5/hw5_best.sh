#!/bin/bash
tar zxvf model.tgz
python3 -B test.py --input $1 --output $2
rm -rf log
