#!/usr/bin/python 

from cmath import exp
import time
import sys
import configparser
import logging
from os import path, mkdir
from time import sleep
from datetime import datetime, tzinfo, timedelta
from ast import literal_eval as le
from glob import glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import tzdata
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pymysql
from xgboost import XGBRegressor as XGBR

feature_dtypes = {'location':'int64','lead_minutes':'int64','timestamp':'int64','stch.height':'float64','stch.saa':'float64','cf1':'float64','R_mean1':'float64','G_mean1':'float64','B_mean1':'float64','R_min1':'int64','G_min1':'int64','B_min1':'int64','R_max1':'int64','G_max1':'int64','B_max1':'int64','RBR1':'float64','cf2':'float64','R_mean2':'float64','G_mean2':'float64','B_mean2':'float64','R_min2':'int64','G_min2':'int64','B_min2':'int64','R_max2':'int64','G_max2':'int64','B_max2':'int64','RBR2':'float64','cf_total':'float64','velocity_x':'float64','velocity_y':'float64','max_ghi':'float64','max_dni':'float64','max_dhi':'float64','forecast_time':'int64','measured_ghi':'float64'}

def readConfig():
    try:
        try:
            config_path = sys.argv[1]
        except Exception:
            config_path = "./config.conf"
        cp = configparser.ConfigParser()
        cp.read(config_path)

        try:
            dbConnection = {}
            dbConnection["user"] = cp["connectionInfoReader"]["user"]
            dbConnection["password"] = cp["connectionInfoReader"]["password"]
            dbConnection["host"] = cp["connectionInfoReader"]["host"]
            dbConnection["database"] = cp["connectionInfoReader"]["database"]

            # sfaConnection = {}
            # sfaConnection["user"] = cp["sfaInfo"]["user"]
            # sfaConnection["password"] = cp["sfaInfo"]["password"]
            # sfaConnection["cachedToken"] = cp["sfaInfo"]["cachedToken"]
            # sfaConnection["maxTries"] = cp["sfaInfo"]["maxTries"]

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
            forecast_cfg["load_models"] = le(cp["forecast"]["load_models"])
            forecast_cfg["model_files"] = le(cp["forecast"]["model_files"])
            forecast_cfg["models_path"] = le(cp["paths"]["models_path"])
            
            ghi_cfg = {}
            ghi_cfg["timezone"] = cp["GHI_sensors"]["GHI_timezone"]
            ghi_cfg["overwriteGHI"] = le(cp["GHI_sensors"]["overwriteGHI"])

        except KeyError as e:
            logging.error("Missing config information: " + str(e))
            return
        
        return {"db": dbConnection, "features": feature_cfg, "ghi": ghi_cfg, "forecast": forecast_cfg}

    except Exception as e:
        logging.error("Error loading config: " + str(e))


def chunk(df, n):
    #Credit for the quick one-liner: https://stackoverflow.com/questions/44729727/pandas-slice-large-dataframe-in-chunks  
    return {i+min(n, len(df[i:i+n])):df[i:i+n] for i in range(0,df.shape[0],n)}


def init_models(fpath, model_files):
    logging.info("Loading models:")
    models = {}
    for lead_time, m in model_files.items():
        logging.info("\t" + m)
        try:
            models[lead_time] = XGBR()
            models[lead_time].load_model(path.join(fpath, m))
        except Exception as e:
            logging.warning("Error loading " + m + ": " + str(e))

    logging.info("Loaded " + str(len(models)) + " models.")
    return models

def calcForecast(df, measured_ghi, config_id=None, models=None):
    #forecast_time = df["timestamp"] + df["lead_minutes"]*60
    df['forecast_time'] = df["timestamp"] + df["lead_minutes"]*60
    output_fields = ["timestamp", "forecast_time", "lead_minutes", "location", "value"]

    if config_id=="static_eq":
        #=(((COS(PI()*0.5*MIN(1,K2)))+(COS(PI()*0.5*MIN(1,J2))))/2*L2)+((SIN(PI()*K2))*M2)  
        out = (((np.cos(np.pi*0.5*np.minimum(1,df["cf2"].astype(float))))+(np.cos(np.pi*0.5*np.minimum(1,df["cf_total"].astype(float)))))/2*df["max_ghi"])+((np.sin(np.pi*df["cf_total"].astype(float)))*df["max_dhi"])
        out_df = pd.concat([df["timestamp"], df["forecast_time"], df["lead_minutes"], df["location"], out], axis=1, keys=output_fields)

    elif config_id=="xgboost_211220":
        input_fields = ['cf1', 'R_mean1', 'G_mean1', 'B_mean1', 'R_min1', 'G_min1', 'B_min1',
       'R_max1', 'G_max1', 'B_max1', 'RBR1', 'cf2', 'R_mean2', 'G_mean2',
       'B_mean2', 'R_min2', 'G_min2', 'B_min2', 'R_max2', 'G_max2', 'B_max2',
       'RBR2', 'max_ghi','cf_total','measured_ghi']

        df = pd.merge(df, measured_ghi, left_on=df["timestamp"], right_index=True, how="left")
        df.rename({location_id: "measured_ghi"}, axis=1, inplace=True)
        df["measured_ghi"] /= df["max_ghi"]

        out_df = pd.DataFrame()
        for lead_mintues, input_df in df.groupby("lead_minutes"):
            predict_ser = pd.Series(models[lead_mintues].predict(input_df[input_fields]), index=input_df.index)
            predict_ser *= input_df['max_ghi']
            predict_df = input_df[output_fields[:-1]]
            predict_df["value"] = predict_ser
            out_df = out_df.append(predict_df)

    else:   #default to persistence
        if measured_ghi is None:
            logging.warning("Missing GHI data, skipping persistence calc")
            out = None
        else:
            out = calcSmartPersist(df, measured_ghi)
        
        out_df = pd.concat([df["timestamp"], df["forecast_time"], df["lead_minutes"], df["location"], out], axis=1, keys=output_fields)

    #out_df.insert(2, "config_id", config_id)
    
    return out_df


def calcSmartPersist(df, measured_ghi):
    #merge actual GHI on base times, then output by forecast_time, effectively projecting current GHI to the forecast time
    persist_df = pd.merge(df, measured_ghi, left_on=df["timestamp"], right_index=True, how="left")
    persist_df.rename({location_id: "measured_ghi"}, axis=1, inplace=True)

    out = persist_df['measured_ghi']
    return out


def calcValidation(features, measured_ghi, location_id):
    try:
        features = pd.merge(features, measured_ghi[location_id], left_on=features["forecast_time"], right_index=True, how="left")
        features.rename({location_id: "measured_ghi"}, axis=1, inplace=True)

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


def plot_stats(maes, mses, title, fpath, use_roots=True):

    title_suffix = " MSE"
    for mse in mses:
        if use_roots:
            mse["mse"] = np.sqrt(mse["mse"])
            title_suffix = " RMSE"
        plt.plot(mse["lead_minutes"], mse["mse"], marker=mse["marker"], linestyle='solid', label=mse["label"])
    plt.title(title + title_suffix)
    plt.legend()
    plt.savefig(path.join(fpath, title + "_RMSE.png"))
    plt.close()

    for mae in maes:
        plt.plot(mae["lead_minutes"], mae["mae"], marker=mae["marker"], linestyle='solid', label=mae["label"])
    plt.title(title + " MAE")
    plt.legend()
    plt.savefig(path.join(config["forecast"]["path"], day, day + "_" + title + "_MAE.png"))
    plt.close()

    return


if __name__ == "__main__":
    startdt = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logFile = "Log_getFeatureGHI_" + startdt + ".log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s', handlers=[logging.FileHandler(logFile), logging.StreamHandler()])
    config = readConfig()

    #Get nowcast metadata
    #db_engine = DBengine(config["db"])
    #sites = db_engine.getSites()
    #sensors = db_engine.getSensors()

    #Load models
    #try:
    if config["forecast"]["load_models"]:
        models = init_models(config["forecast"]["models_path"], config["forecast"]["model_files"])
    #except Exception:
    #    pass

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
            if not path.isdir(config["forecast"]["path"]):
                try:
                    logging.info("  Creating directory " + config["forecast"]["path"])
                    mkdir(config["forecast"]["path"])
                except OSError as e:
                    logging.info("  Error creating directory " + config["forecast"]["path"] + ": " + str(e))
                    continue

            if not path.isdir(dir):
                try:
                    logging.info("  Creating directory " + dir)
                    mkdir(dir)
                except OSError as e:
                    logging.info("  Error creating directory " + dir + ": " + str(e))
                    continue

            #Load GHI data for validation
            try:
                measured_ghi_df = pd.read_csv(path.join(config["features"]["path"], day, "Measured_GHI_1sec.csv"), index_col="timestamp")
                skip_val = False
            except FileNotFoundError:
                logging.warning("GHI file not found, skipping validation.")
                skip_val = True

            for f in flist:
                #Load feature files
                logging.info("\tReading " + f)
                try:
                    f_features = pd.read_csv(f, na_values=[' nan'])
                    dtypes = {k:f for k,f in feature_dtypes.items() if k in f_features.columns}
                    f_features.dropna(inplace=True)
                    f_features = f_features.astype(dtypes)
                except pd.errors.EmptyDataError as e:
                    logging.warning("\t  No feature file found, skipping.  (%s)" % (str(e)))
                    continue
                try:
                    location_id = str(f_features["location"].iloc[0])
                except (KeyError, IndexError) as e:
                    logging.warning("\t  Invalid feature file format (probably from an older version missing the location column), skipping.  (%s)" % (str(e)))
                    continue

                try:
                    #Calc forecast
                    forecast_df = calcForecast(f_features, measured_ghi_df[location_id], config["forecast"]["config_id"], models=models)  

                    #Calc persistence
                    persist_df = calcForecast(f_features, measured_ghi_df[location_id], config_id="persistence")
                except KeyError as e:
                    logging.warning("Missing data for location " + location_id + " ("+ str(e) + "), skipping") #  Trying again without actual GHI, can't calc persistence or stats.")
                    continue
                    # try:
                    #     forecast_df = calcForecast(f_features, None, config["forecast"]["config_id"])  
                    #     persist_df = calcForecast(f_features, None, config_id="persistence")
                    # except Exception as e:
                    #    logging.warning("Unable to generate forecast for location " + location_id + ": " + str(e))

                #Save forecast
                try:
                    out_path_day = path.join(config["forecast"]["path"], day, f[f.find("GHI"):-4])
                    out_path_base = path.join(config["forecast"]["path"], days[0] + "_" + days[-1] + "_" + f[f.find("GHI"):-4])
                    out_path_forecast = "_forecast_" + config["forecast"]["config_id"] + "_"
                    
                    for lead_mintues, grouped_df in forecast_df.groupby("lead_minutes"):
                        grouped_df.to_csv(out_path_day + out_path_forecast + str(lead_mintues) + ".csv", index=False) 
                        grouped_df.to_csv(out_path_base + out_path_forecast + str(lead_mintues) + "_total.csv", index=False, mode='a', header=not path.exists(out_path_base + out_path_forecast + str(lead_mintues) + "_total.csv")) 
                    
                    for lead_mintues, grouped_df in persist_df.groupby("lead_minutes"):
                        grouped_df.to_csv(out_path_day + "_forecast_persist_" + str(lead_mintues) + ".csv", index=False)        #mostly for debugging/validation
                        grouped_df.to_csv(out_path_base + "_forecast_persist_" + str(lead_mintues) + "_total.csv", index=False, mode='a', header=not path.exists(out_path_base + "_persist_" + str(lead_mintues) + "_total.csv"))        #mostly for debugging/validation
                except Exception as e:
                    logging.error("Error saving forecast for " + day + ": " + str(e))         

                #Calc forecast validation
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