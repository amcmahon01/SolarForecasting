import sys
from os import path
import socket
from ast import literal_eval
from configparser import ConfigParser

host = socket.gethostname()

try:
    print("Saying hello from " + host)
    config_file = "./config_" + host + ".conf"
    if not path.exists(config_file):
        raise
except Exception:
    config_file = "./dist_config.conf"

# Read the configuration file
print("Reading the configuration file " + config_file)
cp = ConfigParser()
cp.read(config_file)

data_path =  literal_eval(cp["paths"]["data_root"])
temp_path =  literal_eval(cp["paths"]["temp_root"])
output_path =  literal_eval(cp["paths"]["output_root"])

print("\tdata path:" + data_path)
print("\ttemp (local) path:" + temp_path)
print("\toutput path:" + output_path)

with open(path.join(output_path, "hello_from_"+host+".txt"), "w") as f:
    f.write("Hi! My paths are:\ndata: "+data_path+"\ntemp(local): "+temp_path+"\noutput: "+output_path)