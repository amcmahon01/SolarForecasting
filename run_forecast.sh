#!/bin/bash
HDF5_USE_FILE_LOCKING=FALSE
python preprocess.py
python generate_stitch.py
python extract_features.py
#python predict.py