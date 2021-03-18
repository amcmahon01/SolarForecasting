import sys
import os
from os import path
from ast import literal_eval
from datetime import datetime
from configparser import ConfigParser


def read_dist_config():
    #Load distributed config
    try:
        dist_config_file = "./distributed/dist_config.conf"

        # Read the configuration file
        print("Reading the distributed configuration file: " + dist_config_file)
        cp_dist = ConfigParser()
        cp_dist.read(dist_config_file)

        print("\tdata path: " + cp_dist["paths"]["data_root"])
        print("\ttemp (local) path: " + cp_dist["paths"]["temp_root"])
        print("\toutput path: " + cp_dist["paths"]["output_root"])

        return cp_dist

    except Exception as e:
        print("Error loading distributed config, cannot continue!\n " + str(e))
        exit()


def read_base_config():
    #Load forecasting config file
    try:
        forecast_config_file = "./config.conf"
        # Read the configuration file
        print("Reading the forecast configuration file: " + forecast_config_file)
        cp_forecast = ConfigParser()
        cp_forecast.read(forecast_config_file)
        return cp_forecast

    except Exception as e:
        print("Error loading forecast config, cannot continue!\n " + str(e))
        exit()


if __name__ == "__main__":
    dist_config = read_dist_config()

    start_date = datetime.strptime(dist_config["forecast"]["start_date"], '%Y-%m-%d')
    end_date = datetime.strptime(dist_config["forecast"]["end_date"], '%Y-%m-%d')
    day_list = [datetime.fromordinal(i).strftime("%Y-%m-%d") for i in range(start_date.toordinal(), end_date.toordinal()+1)]

    print("Distributed config:\n\tdays: " + str(len(day_list)) + "\n\tstart date: " + day_list[0] + "\n\tend date: " + day_list[-1])


    hosts = {}
    day_index = 0
    
    #Load hosts and work distribution
    try:
        with open("./distributed/dist_hosts.csv", "r") as hosts_file:
            for l in hosts_file.readlines():
                try:
                    host, cores, day_count = l.rstrip().split(',')

                    day_count = int(day_count)
                    if day_index + day_count > len(day_list)-1:
                        day_count = (len(day_list) - 1) - day_index   #in case too many days are assigned

                    #Read "base" forecast config
                    base_config = read_base_config()        #there's definitely a more efficient way to do this than loading every time, but deepcopy has some quirks
                
                    #Update paths
                    base_config.set("paths","inpath","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["inpath"])+"'")
                    base_config.set("paths","raw_GHI_path","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["raw_GHI_path"])+"'")
                    base_config.set("paths","static_mask_path","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["static_mask_path"])+"'")
                    base_config.set("paths","tmpfs","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["tmpfs"])+"'")
                    base_config.set("paths","GHI_path","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["GHI_path"])+"'")
                    base_config.set("paths","stitch_path","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["stitch_path"])+"'")
                    base_config.set("paths","feature_path","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["feature_path"])+"'")
                    base_config.set("paths","forecast_path","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["forecast_path"])+"'")
                    base_config.set("paths","stats_path","'"+path.join(dist_config["paths"]["data_root"], dist_config["paths"]["stats_path"])+"'")

                    #Update cores
                    base_config.set("server","cores_to_use",cores)

                    #Update date range
                    base_config.set("forecast","days",str(day_list[day_index:day_index+day_count-1]))

                    #Save host config
                    with open("config_"+host+".conf","w") as host_conf_file:
                        base_config.write(host_conf_file)

                    print(host + "\n\tcores: " + cores + "\n\tdays: " + str(day_count) + "\n\tstart date: " + day_list[day_index] + "\n\tend date: " + day_list[day_index+day_count-1])
                    day_index += day_count

                except Exception as e:
                    print("Error processing "+host+": " + str(e))

    except Exception as e:
        print("Error loading hosts, cannot continue!\n " + str(e))

