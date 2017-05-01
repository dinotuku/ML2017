#!/bin/bash
cat ./model/strong_epoch0_G_0/best_model.hdf5.tar.* > ./model/strong_epoch0_G_0/best_model.hdf5.tar
tar -xvf ./model/strong_epoch0_G_0/best_model.hdf5.tar -C ./model/strong_epoch0_G_0/
python3 -B test.py --input $1 --output $2 --model strong --epoch 0 --dataGen True --idx 0 --choice best
rm -rf ./model/strong_epoch0_G_0/best_model.hdf5.tar ./model/strong_epoch0_G_0/best_model.hdf5
