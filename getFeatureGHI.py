#!/usr/bin/python 

import time
import sys
import configparser
import logging
from os import path
from time import sleep
from datetime import datetime, tzinfo, timedelta
from ast import literal_eval as le
from glob import glob
import tzdata
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
import pandas as pd
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
            dbConenction["user"] = cp["connectionInfoReader"]["user"]
            dbConenction["password"] = cp["connectionInfoReader"]["password"]
            dbConenction["host"] = cp["connectionInfoReader"]["host"]
            dbConenction["database"] = cp["connectionInfoReader"]["database"]

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
            
            ghi_cfg = {}
            ghi_cfg["timezone"] = cp["GHI_sensors"]["GHI_timezone"]
            ghi_cfg["overwriteGHI"] = le(cp["GHI_sensors"]["overwriteGHI"])

        except KeyError as e:
            logging.error("Missing config information: " + str(e))
            return
        
        return {"db": dbConenction, "features": feature_cfg, "ghi": ghi_cfg}

    except Exception as e:
        logging.error("Error loading config: " + str(e))



def chunk(df, n):
    #Credit for the quick one-liner: https://stackoverflow.com/questions/44729727/pandas-slice-large-dataframe-in-chunks  
    return {i+min(n, len(df[i:i+n])):df[i:i+n] for i in range(0,df.shape[0],n)}


class DBengine():

    def __init__(self, config):
        logging.info("Connecting to " + config["host"] + "/"+ config["database"])
        self.engine = create_engine("mysql+pymysql://" + config["user"] + ":" + config["password"] + "@" + config["host"] + "/"+ config["database"] + "?charset=utf8mb4&autocommit=true")

    def getSites(self):
        try:
            sites = pd.read_sql_table('sites', self.engine)
            site_d = sites.to_dict('records')
            return site_d
        except Exception as e:
            logging.error("Error getting sites: " + str(e))
            raise e

    def getSensors(self):
        try:
            sensors = pd.read_sql('SELECT sensors.sensor_id, sensors.site_id, sensors.measurement, sensors.instrument, sensors.units, sites.name AS site_name FROM sensors INNER JOIN sites ON sensors.site_id = sites.site_id', self.engine)
            sensors_d = sensors.to_dict('records')
        except Exception as e:
            logging.error("Error getting sensor records: " + str(e))
            return {}
        return sensors_d

    def getData(self, sensor, exclude_range=True, table="observations_1min_avg"):
        logging.info("Getting data for " + sensor["obs_name"])
        try:    #Check for exclusion range  ****FUTURE: Add "value gaps" check
            start_dt = sensor["start_dt"].tz_convert("EST").strftime("%Y-%m-%d %H:%M:%S")
            end_dt = sensor["end_dt"].tz_convert("EST").strftime("%Y-%m-%d %H:%M:%S")
            if exclude_range:
                q_dt = " AND NOT (TIMESTAMP BETWEEN '" + start_dt + "' AND '" + end_dt + "')"
            else:
                q_dt = " AND (TIMESTAMP BETWEEN '" + start_dt + "' AND '" + end_dt + "')"
        except (KeyError, ValueError):
            q_dt = ""
        data = pd.read_sql('SELECT TIMESTAMP, value, quality_flag FROM ' + table + ' WHERE sensor_id=' + str(sensor["sensor_id"]) + q_dt, self.engine)
        data.set_index(data["TIMESTAMP"], inplace=True)
        data.drop(columns="TIMESTAMP", inplace=True)
        data = data.tz_localize("EST")
        logging.info("\tRecords found: " + str(len(data)))
        return data
        
        
    def getFeatureGHI(self, sensor, table="observations_1sec", intSecs=30):
        #Gets all GHI values for the day, starting from start_dt and incrementing by intSecs.  Assumes even intervals rather than running 1000+ queries per day.
        logging.info("Getting data for " + sensor["obs_name"])
        try:    #Check for exclusion range  ****FUTURE: Add "value gaps" check
            start_dt = datetime.fromtimestamp(sensor["start_dt"], tz=ZoneInfo('UTC')).strftime("%Y-%m-%d %H:%M:%S")
            end_dt = datetime.fromtimestamp(sensor["end_dt"], tz=ZoneInfo('UTC')).strftime("%Y-%m-%d %H:%M:%S")

            q_dt = " AND (TIMESTAMP BETWEEN '" + start_dt + "' AND '" + end_dt + "') AND (((SECOND(TIMESTAMP) - SECOND('" + start_dt + "')) % " + str(intSecs) + ") = 0) LIMIT 2880";

        except (KeyError, ValueError) as e:
            logging.error("Missing key parameters: " + str(e))
            q_dt = ""
            return None

        data = pd.read_sql('SELECT TIMESTAMP, value FROM ' + table + ' WHERE sensor_id=' + str(sensor["sensor_id"]) + q_dt, self.engine)
        data.set_index(data["TIMESTAMP"], inplace=True)
        data.drop(columns="TIMESTAMP", inplace=True)
        data = data.tz_localize("EST")
        logging.info("\tRecords found: " + str(len(data)))
        return data


    def getAllFeatureGHI(self, sensors, table="observations_1sec", intSecs=30, subsample=True):
        #Gets all GHI values for the day, starting from start_dt and incrementing by intSecs.  Assumes even intervals rather than running 1000+ queries per day.
        logging.info("Getting data for " + str(len(sensors)) + " sensors...")
        try:    #Check for exclusion range  ****FUTURE: Add "value gaps" check

            start_dt_min = min([i["start_dt"] for i in sensors if "start_dt" in i.keys()])
            end_dt_max = max([i["end_dt"] for i in sensors if "end_dt" in i.keys()])

            start_dt = datetime.fromtimestamp(start_dt_min, tz=ZoneInfo(config["ghi"]["timezone"])).strftime("%Y-%m-%d %H:%M:%S")
            end_dt = datetime.fromtimestamp(end_dt_max, tz=ZoneInfo(config["ghi"]["timezone"])).strftime("%Y-%m-%d %H:%M:%S")

            if subsample:
                #Note double % to escape python string formatter
                q_dt = "(TIMESTAMP BETWEEN '" + start_dt + "' AND '" + end_dt + "') AND (((SECOND(TIMESTAMP) - SECOND('" + start_dt + "')) %% " + str(intSecs) + ") = 0) LIMIT 288000"
            else:
                q_dt = "(TIMESTAMP BETWEEN '" + start_dt + "' AND '" + end_dt + "') LIMIT 2000000"

        except (KeyError, ValueError) as e:
            logging.error("Missing key parameters: " + str(e))
            q_dt = ""
            return None

        data = pd.read_sql('SELECT TIMESTAMP, sensor_id, value FROM ' + table + ' WHERE ' + q_dt, self.engine)
        data.set_index(data["TIMESTAMP"], inplace=True)
        data.drop(columns="TIMESTAMP", inplace=True)
        data = data.tz_localize("EST")
        g = data.groupby("sensor_id")
        data = pd.DataFrame({idx - 1: g.get_group(idx)["value"] for idx in g.groups.keys()})    #Note "-1" on index to match feature extraction output
        logging.info("\tRecords found: " + str(len(data)))
        if len(g.groups.keys()) < len(sensors):
            logging.warning("Missing GHI data for %i sensors" % (len(sensors) - len(g.groups.keys())))
        return data




if __name__ == "__main__":
    startdt = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logFile = "Log_getFeatureGHI_" + startdt + ".log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s', handlers=[logging.FileHandler(logFile), logging.StreamHandler()])
    config = readConfig()

    #Get nowcast metadata
    db_engine = DBengine(config["db"])
    #sites = db_engine.getSites()
    sensors = db_engine.getSensors()

    #Get days
    fpath = path.join(config["features"]["path"], "*")
    logging.info("Using feature path " + fpath)
    try:
        days = config["features"]["days"]
    except Exception:
        days = sorted(glob(fpath))
        days = [path.split(d)[-1] for d in days if path.isdir(d)]     # Keep only dirs

    for day in days:
        logging.info("Loading data for " + day)
        if not path.isdir(path.join(config["features"]["path"], day)):
            logging.info("  No features, skipping.")
            continue
        try:    
            if config["ghi"]["overwriteGHI"]:
                logging.info("  Overwriting existing GHI files")
            else:
                raise Exception
        except Exception:
            if path.isfile(path.join(config["features"]["path"], day, "Measured_GHI_1sec.csv")) and path.isfile(path.join(config["features"]["path"], day, "Measured_GHI_1min_avg.csv")):
                logging.info("  Outputs already exist, skipping.")
                continue

        #Get feature data
        flist = sorted(glob(path.join(config["features"]["path"], day, "GHI*.csv")))

        for f in flist:
            logging.info("\tReading " + f)
            try:
                f_features = pd.read_csv(f)
            except pd.errors.EmptyDataError as e:
                logging.warning("\t  No feature file found, skipping.  (%s)" % (str(e)))
                continue
                
            try:
                start_dt = f_features["timestamp"].iloc[0]
                end_dt = max(f_features["timestamp"] + (f_features["lead_minutes"]*60))
                sensor_id = int(f_features["location"][0]) + 1                
            except IndexError as e:
                logging.warning("\t  No feature data found, skipping.  (%s)" % (str(e)))
                continue
            except KeyError as e:
                logging.warning("\t  Invalid feature file format (probably from an older version missing the location column), skipping.  (%s)" % (str(e)))
                continue

            try:
                obs_name = sensors[sensor_id - 1]["site_name"] + " " + sensors[sensor_id - 1]["measurement"]    #Try to use existing db info
            except KeyError:
                obs_name = "GHI Location " + str(int(f_features["location"][0]) + 1)                            #Fall back on generic location name

            sensors[sensor_id - 1].update({"start_dt":start_dt, "end_dt": end_dt, "sensor_id": sensor_id, "obs_name":obs_name})
        
        #Get 1 sec data
        try:
            feat_GHI = db_engine.getAllFeatureGHI(sensors, subsample=False)
            feat_GHI.set_index(feat_GHI.index)
            feat_GHI['timestamp'] = feat_GHI.index.view('int64')//1e9
            feat_GHI.set_index(feat_GHI["timestamp"], inplace=True)
            feat_GHI.drop(columns="timestamp", inplace=True)
            feat_GHI.to_csv(path.join(config["features"]["path"], day, "Measured_GHI_1sec.csv"))
        except Exception as e:
            logging.error("  Error loading/saving 1 sec data: " + str(e))
        
        #Get 1 min avg data
        try:
            feat_GHI = db_engine.getAllFeatureGHI(sensors, table="observations_1min_avg", subsample=False)
            feat_GHI.set_index(feat_GHI.index)
            feat_GHI['timestamp'] = feat_GHI.index.view('int64')//1e9
            feat_GHI.set_index(feat_GHI["timestamp"], inplace=True)
            feat_GHI.drop(columns="timestamp", inplace=True)
            feat_GHI.to_csv(path.join(config["features"]["path"], day, "Measured_GHI_1min_avg.csv"))    
        except Exception as e:
            logging.error("  Error loading/saving 1 min avg data: " + str(e))        

    logging.info("Done updating data.")