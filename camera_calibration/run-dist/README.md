# Run-time Configuration #

This directory contains the sample run-time configuration files for all the `camera_calibration` modules.

## `*.conf` ##

Contains the input/output file-paths and the various module configuration parameters. They are in `config` format and are parsed by the python module `configparser`. One such configuration file is required as the first positional command-line argument.

## `*.yaml` ##

The `.yaml` files store the parameters that describe the optical characteristics of the cameras and their orientation. The output of the least-squares fit are stored in two different formats. The main input/output file uses named parameters in a python dict and is defined in the `.conf` file by the `camera_cal_file_optimized` parameter. Save a copy of the initial `camera_cal_file_optimized` in the file specified by the `camera_cal_file` config parameter and you can show the improvement with `plotAllResiduals.py`. The `camera_cal_file_list` parameter specifies the file for the alternate output format where the parameters for each camera are specified as a list in the order expected by the `camera.py` module in the main operational branch.

```
[ nx0,cy,cx,rot,beta,azm,c1,c2,c3]
```

The `.yaml` files are read and parsed by the `yaml` module.

