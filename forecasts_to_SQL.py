#!/usr/bin/python 

import sys
import configparser
import logging
from os import path
from glob import glob
from datetime import datetime
from ast import literal_eval as le

import db as sqldb

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
            dbConnection["user"] = cp["connectionInfoWriter"]["user"]
            dbConnection["password"] = cp["connectionInfoWriter"]["password"]
            dbConnection["host"] = cp["connectionInfoWriter"]["host"]
            dbConnection["database"] = cp["connectionInfoWriter"]["database"]
            dbConnection["table"] = cp["connectionInfoWriter"]["table"]

            forecast_cfg = {}
            forecast_cfg["path"] = le(cp["paths"]["forecast_path"])
            forecast_cfg["config_id"] = cp["forecast"]["config_id"]

        except KeyError as e:
            logging.error("Missing config information: " + str(e))
            return
        
        return {"connectionInfo": dbConnection, "forecast":forecast_cfg}

    except Exception as e:
        logging.error("Error loading config: " + str(e))



if __name__ == "__main__":
    startdt = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    try:
        logFile = "Log_ForecastToSQL_" + startdt + "_" + path.splitext(path.basename(sys.argv[1]))[0] + ".log"
    except Exception:
        logFile = "Log_ForecastToSQL_" + startdt + ".log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler(logFile), logging.StreamHandler()])
    config = readConfig()
    
    try:
        db = sqldb.DBengine(config["connectionInfo"])

        logging.info("Using feature path " + config["forecast"]["path"])
        flist = sorted(glob(path.join(config["forecast"]["path"], "*_total.csv")))
        logging.info("\tFound %i files." % len(flist))

        for f_name in flist:
            try:
                logging.info("File: " + f_name)

                try:
                    forecast_id = db.createForecast(f_name)
                except Exception as e:
                    logging.warning("Error creating forecast: " + str(e))

                try:
                    db.loadForecastData(f_name, forecast_id)
                except Exception as e:
                    logging.warning("Error loading data for id " + str(forecast_id) + ": " + str(e))

            except Exception as e:
                logging.error("Error processsing " + f_name + ": " + str(e))
        
        logging.info("Done, waiting for file updates...")

    finally:
        db.engine.dispose()