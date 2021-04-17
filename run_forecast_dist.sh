#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
echo "$HOSTNAME"
config_path="./distributed/config/config_$HOSTNAME.conf"
python preprocess.py $config_path
python generate_stitch.py $config_path
python extract_features.py $config_path
#python predict.py $config_path