#!/usr/bin/python 

import time
import sys
import configparser
import logging
from os import path, mkdir
from time import sleep
from datetime import datetime, tzinfo, timedelta
from ast import literal_eval as le
from glob import glob
import tzdata
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pymysql


def readConfig():
    try:
        try:
            config_path = sys.argv[1]
        except Exception:
            config_path = "./config.conf"
        cp = configparser.ConfigParser()
        cp.read(config_path)

        try:
            dbConenction = {}
            # dbConenction["user"] = cp["connectionInfoWriter"]["user"]
            # dbConenction["password"] = cp["connectionInfoWriter"]["password"]
            # dbConenction["host"] = cp["connectionInfoWriter"]["host"]
            # dbConenction["database"] = cp["connectionInfoWriter"]["database"]

            feature_cfg = {}
            feature_cfg["path"] = le(cp["paths"]["feature_path"])
            try:
                feature_cfg["start_date"] = datetime.strptime(le(cp["forecast"]["start_date"]), "%Y%m%d")
                feature_cfg["end_date"] = datetime.strptime(le(cp["forecast"]["end_date"]), "%Y%m%d")
                feature_cfg["days"] = [y.strftime('%Y%m%d') for y in [feature_cfg["start_date"]+timedelta(days=x) for x in range((feature_cfg["end_date"]-feature_cfg["start_date"]).days + 1)]]
            except KeyError:
                try:
                    feature_cfg["days"] = le(cp["forecast"]["days"])
                except KeyError:
                    logging.info("No date range specified, using all available")

            forecast_cfg = {}
            forecast_cfg["path"] = le(cp["paths"]["forecast_path"])
            forecast_cfg["config_id"] = cp["forecast"]["config_id"]
            
            ghi_cfg = {}
            ghi_cfg["timezone"] = cp["GHI_sensors"]["GHI_timezone"]
            ghi_cfg["overwriteGHI"] = le(cp["GHI_sensors"]["overwriteGHI"])

        except KeyError as e:
            logging.error("Missing config information: " + str(e))
            return
        
        return {"db": dbConenction, "features": feature_cfg, "ghi": ghi_cfg, "forecast": forecast_cfg}

    except Exception as e:
        logging.error("Error loading config: " + str(e))


def chunk(df, n):
    #Credit for the quick one-liner: https://stackoverflow.com/questions/44729727/pandas-slice-large-dataframe-in-chunks  
    return {i+min(n, len(df[i:i+n])):df[i:i+n] for i in range(0,df.shape[0],n)}


def calcForecast(df, config_id=0):
    forecast_time = df["timestamp"] + df["lead_minutes"]
    #=(((COS(PI()*0.5*MIN(1,K2)))+(COS(PI()*0.5*MIN(1,J2))))/2*L2)+((SIN(PI()*K2))*M2)  
    out = (((np.cos(np.pi*0.5*np.minimum(1,df["cf2"].astype(float))))+(np.cos(np.pi*0.5*np.minimum(1,df["cf_total"].astype(float)))))/2*df["max_ghi"])+((np.sin(np.pi*df["cf_total"].astype(float)))*df["max_dhi"])

    keys = ["timestamp", "forecast_time", "lead_minutes", "location", "value"]
    out_df = pd.concat([df["timestamp"], forecast_time, df["lead_minutes"], df["location"], out], axis=1, keys=keys)
    out_df.insert(2, "config_id", config_id)
    
    return out_df


def calcSmartPersist(df):
    forecast_time = df["timestamp"] + df["lead_minutes"]



def calcValidation(features, measured_ghi, location_id):
    try:
        features = pd.merge(features, measured_ghi.set_index("timestamp")[str(location_id)], left_on=features["forecast_time"], right_index=True, how="left")
        features.rename({str(location_id): "measured_ghi"}, axis=1, inplace=True)

        features["abs_diff"] = np.abs(features["measured_ghi"] - features["value"])
        features["sqr_diff"] = np.square(features["measured_ghi"] - features["value"])
        f_groups = features.groupby("lead_minutes")

        mae = f_groups["abs_diff"].mean()
        mse = f_groups["sqr_diff"].mean()
        n = f_groups["abs_diff"].count()    #should match sqr_diff

        features.drop("abs_diff", axis=1, inplace=True)
        features.drop("sqr_diff", axis=1, inplace=True)

        return features, mae, mse, n

    except KeyError as e:
        logging.warning("\t\tMissing data:" + str(e))
        f_groups = features.groupby("lead_minutes")
        return features, pd.Series([np.NaN]*len(f_groups), index=f_groups.groups.keys()), pd.Series([np.NaN]*len(f_groups), index=f_groups.groups.keys()), pd.Series([0]*len(f_groups), index=f_groups.groups.keys())


def calcTotalStats(stats_list):
    #Save total validation stats
    total_val_df = pd.DataFrame(stats_list)
    lead_steps = int((len(total_val_df.columns) - 2)/3)      #don't include day or location_id

    #Ugly but functional...
    n_total = {}
    mae_total = {}
    mse_total = {}
    stat_totals = ["total", -1]
    for i in range(0, lead_steps):
        lead = total_val_df.columns[(i*3)+4].split('_')[-1]
        n_total[lead] = np.sum(total_val_df.iloc[:,(i*3)+4])
        mae_total[lead] = np.sum(total_val_df.iloc[:,(i*3)+2] * total_val_df.iloc[:,(i*3)+4])/n_total[lead]
        mse_total[lead] = np.sum(total_val_df.iloc[:,(i*3)+3] * total_val_df.iloc[:,(i*3)+4])/n_total[lead]
        stat_totals += [mae_total[lead], mse_total[lead], n_total[lead]]
    try:
        total_val_df.loc["totals"] = stat_totals
        logging.info({"mae":mae_total, "mse":mse_total, "n":n_total})
    except ValueError as e:
        logging.warning("No stats calculated: " + str(e))

    return total_val_df


if __name__ == "__main__":
    startdt = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logFile = "Log_getFeatureGHI_" + startdt + ".log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s', handlers=[logging.FileHandler(logFile), logging.StreamHandler()])
    config = readConfig()

    #Get nowcast metadata
    #db_engine = DBengine(config["db"])
    #sites = db_engine.getSites()
    #sensors = db_engine.getSensors()

    #Get days
    fpath = path.join(config["features"]["path"], "*")
    logging.info("Using feature path " + fpath)
    try:
        days = config["features"]["days"]
    except Exception:
        days = sorted(glob(fpath))
        days = [path.split(d)[-1] for d in days if path.isdir(d)]     # Keep only dirs
    total_val_l = []

    for day in days:
        logging.info("Loading data for " + day)
        daily_val_l = []

        #Get feature file list
        flist = sorted(glob(path.join(config["features"]["path"], day, "GHI*.csv")))

        if len(flist) > 0:
            dir = config["forecast"]["path"] + day[:8]
            if not path.isdir(dir):
                try:
                    logging.info("  Creating directory " + dir)
                    mkdir(dir)
                except OSError as e:
                    logging.info("  Error creating directory " + dir + ": " + str(e))
                    continue

            #Load GHI data for validation
            try:
                measured_ghi_df = pd.read_csv(path.join(config["features"]["path"], day, "Measured_GHI_1sec.csv"))
                skip_val = False
            except FileNotFoundError:
                logging.warning("GHI file not found, skipping validation.")
                skip_val = True

            for f in flist:
                #Load feature files
                logging.info("\tReading " + f)
                try:
                    f_features = pd.read_csv(f)
                except pd.errors.EmptyDataError as e:
                    logging.warning("\t  No feature file found, skipping.  (%s)" % (str(e)))
                    continue
                try:
                    location_id = f_features["location"][0]
                except (KeyError, IndexError) as e:
                    logging.warning("\t  Invalid feature file format (probably from an older version missing the location column), skipping.  (%s)" % (str(e)))
                    continue

                #Calc forecast
                forecast_df = calcForecast(f_features)   

                #Save forecast
                try:
                    forecast_df.to_csv(path.join(config["forecast"]["path"], day, f[f.find("GHI"):-4] + "_forecast.csv"), index=False)
                except Exception as e:
                    logging.error("Error saving forecast for " + day + ": " + str(e))         

                #Calc validation
                if not skip_val:
                    forecast_df, mae, mse, n = calcValidation(forecast_df, measured_ghi_df, location_id)

                    val_stats = {"day":day, "location_id":location_id}
                    for lead in n.index:
                        val_stats.update({"mae_"+str(lead):mae[lead], "mse_"+str(lead):mse[lead], "n_"+str(lead):n[lead]}) 
                    daily_val_l.append(val_stats)
                    logging.debug(str([day, location_id, mae, mse]))
                
            #Save daily validation stats
            if not skip_val:
                daily_val_df = calcTotalStats(daily_val_l) 
                try:
                    daily_val_df.to_csv(path.join(config["forecast"]["path"], day, day + "_validation.csv"), index=False)
                    daily_dict = dict(daily_val_df.loc["totals"])
                    daily_dict["day"] = day
                    total_val_l += [daily_dict]                
                except Exception as e:
                    logging.error("Error saving stats for " + day + ": " + str(e))

    #Save total validation stats
    total_val_df = calcTotalStats(total_val_l)
    try:
        total_val_df.to_csv(path.join(config["forecast"]["path"], days[0] + "_" + days[-1] + "_validation.csv"), index=False)
    except Exception as e:
        logging.error("Error saving total stats: " + str(e))

    logging.info("Done generating forecast data.")