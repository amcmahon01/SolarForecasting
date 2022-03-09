#!/usr/bin/python 

import time
import sys
import configparser
import logging
from os import path
from time import sleep
from datetime import datetime
from ast import literal_eval as le

import pandas as pd
from db import DBengine

import solarforecastarbiter.datamodel as sfa_dm
from solarforecastarbiter.io import api as sfa_api
from solarforecastarbiter.io import utils as sfa_utils


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

            sfaConnection = {}
            sfaConnection["user"] = cp["sfaInfo"]["user"]
            sfaConnection["password"] = cp["sfaInfo"]["password"]
            sfaConnection["cachedToken"] = cp["sfaInfo"]["cachedToken"]
            sfaConnection["maxTries"] = cp["sfaInfo"]["maxTries"]

        except KeyError as e:
            logging.error("Missing config information: " + str(e))
            return
        
        return {"db": dbConnection, "sfa": sfaConnection}

    except Exception as e:
        logging.error("Error loading config: " + str(e))


def fillMissing(df, freq="1Min", method=None, indexCol="TIMESTAMP", replaceIndexCol=False, useRounding=False):
    logging.info("\tChecking for missings rows...")
    old_len = len(df)
    if useRounding:
        new_idx =  pd.date_range(start=df.index.min().round(freq), end=df.index.max().round(freq), freq=freq)
    else:
        new_idx =  pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)

    df_reindexed = df.reindex(new_idx, method=method, tolerance=pd.Timedelta(freq)/2)

    if replaceIndexCol:
        df_reindexed[indexCol] = df_reindexed.index
    logging.info("\tFilled "+str(len(df_reindexed)-old_len)+" missing rows")
    return df_reindexed

def chunk(df, n):
    #Credit for the quick one-liner: https://stackoverflow.com/questions/44729727/pandas-slice-large-dataframe-in-chunks  
    return {i+min(n, len(df[i:i+n])):df[i:i+n] for i in range(0,df.shape[0],n)}


def syncObs():
    startdt = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logFile = "Log_sfaSync_" + startdt + ".log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s', handlers=[logging.FileHandler(logFile), logging.StreamHandler()])
    if config is None:
        config = readConfig()

    #Get nowcast metadata
    db_engine = DBengine(config["db"])
    sites = db_engine.getSites()
    sensors = db_engine.getSensors()
    forecast_ids = db_engine.getForecastIDs()

    #Get SFA metadata
    sfa_session = SFA(config["sfa"])
    sfa_sites = sfa_session.getSites()
    sfa_obs = sfa_session.listObservations()
    sfa_forecasts = sfa_session.listForecasts()

    site_d = {n.name: n for n in sfa_sites}
    obs_d = {n.name: n for n in sfa_obs}
    

    #Sync sites
    for s in sites:
        if s["name"] not in site_d.keys():                            #Check which records are new    ****FUTURE: Should change to use UUID instead/in addition to****
            site_result = sfa_session.addSite(s)                      #Add missing sites
            db_engine.updateSite(site_result)
            site_d.update({s["name"]:site_result})                    #Add to local site dict

    for sensor in sensors:
        #Sync observation metadata
        sensor["obs_name"] = sensor["site_name"] + " " + sensor["measurement"]  #Site name + measurement should be used for all observation names (ex: "LISF PB1 GHI")
        sensor["site"] = site_d[sensor["site_name"]]                  #Associate SFA site record
        if sensor["obs_name"] not in obs_d.keys():                    #Check which records are new    ****FUTURE: Should change to use UUID instead/in addition to****
            obs_result = sfa_session.addObservation(sensor)           #Add missing observations
            #db_engine.updateObservation(obs_result)                  ****FUTURE: Update local metadata
            obs_d.update({sensor["obs_name"]:obs_result})             #Add to local obs dict

        #Sync obs data
        obs_uuid = obs_d[sensor["obs_name"]].observation_id           #Get observation UUID now that all should be in the local dict
        sensor["start_dt"], sensor["end_dt"] = sfa_session.getObservationTimeRange(obs_uuid)    #Get existing data range (to be excluded from upload)
        data = db_engine.getObsData(sensor)                              #Warning: Could be very large
        filled_data = fillMissing(data)
        total_len = len(filled_data)
        logging.info("Uploading " + str(total_len) + " rows to " + sensor["obs_name"] + " (" + obs_uuid + ")")
        for idx, data_chunk in chunk(filled_data, 129600).items():                 #Create 90 day chunks, should be well below 16MB, 200k row API limits
            sfa_session.addObsData(data_chunk, obs_uuid)              #Add data chunk
            logging.info("\t" + str(round(idx/total_len*100,1)) + "% (" + str(idx) + "/" + str(total_len) + ")")

    logging.info("Done updating data.")


def syncForecasts():
    startdt = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logFile = "Log_sfaSync_" + startdt + ".log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s', handlers=[logging.FileHandler(logFile), logging.StreamHandler()])
    #if config is None:
    config = readConfig()

    #Get nowcast metadata
    db_engine = DBengine(config["db"])
    sites = db_engine.getSites()
    #sensors = db_engine.getSensors()
    forecasts = db_engine.getForecasts()

    #Get SFA metadata
    sfa_session = SFA(config["sfa"])
    sfa_sites = sfa_session.getSites()
    #sfa_obs = sfa_session.listObservations()
    sfa_forecasts = sfa_session.listForecasts()

    site_d = {n.name: n for n in sfa_sites}
    #obs_d = {n.name: n for n in sfa_obs}
    forecasts_d = {n.name: n for n in sfa_forecasts}
    

    #Sync sites
    for s in sites:
        if s["name"] not in site_d.keys():                            #Check which records are new    ****FUTURE: Should change to use UUID instead/in addition to****
            site_result = sfa_session.addSite(s)                      #Add missing sites
            db_engine.updateSite(site_result)
            site_d.update({s["name"]:site_result})                    #Add to local site dict

    for forecast in forecasts:
        #Sync forecast metadata
        try:
            site_name = next(item["name"] for item in sites if item["site_id"] == forecast["site_id"])
        except KeyError:
            logging.warning("Site id " + str(forecast["site_id"]) + " not found, skipping.")
            return

        forecast["forecast_name"] = site_name + " " + str(forecast["lead_time"]) + "min " + forecast["variable"]  #Site name + lead time + measurement should be used for all forecast names (ex: "LISF PB1 10min GHI")
        forecast["site"] = site_d[site_name]                              #Associate SFA site record
        if forecast["forecast_name"] not in forecasts_d.keys():           #Check which records are new    ****FUTURE: Should change to use UUID instead/in addition to****
            result = sfa_session.addForecast(forecast)                    #Add missing forecasts
            #db_engine.updateForecasts(result)                            ****FUTURE: Update local metadata
            forecasts_d.update({forecast["forecast_name"]:result})        #Add to local dict

        #Sync forecast data
        forecast_uuid = forecasts_d[forecast["forecast_name"]].forecast_id                               #Get forecast UUID now that all should be in the local dict
        forecast["start_dt"], forecast["end_dt"] = sfa_session.getForecastTimeRange(forecast_uuid)    #Get existing data range (to be excluded from upload)
        data = db_engine.getForecastData(forecast)                         #Warning: Could be very large
        filled_data = fillMissing(data, method='nearest', useRounding=True)
        total_len = len(filled_data)
        logging.info("Uploading " + str(total_len) + " rows to " + forecast["forecast_name"] + " (" + forecast_uuid + ")")
        for idx, data_chunk in chunk(filled_data, 129600).items():         #Create 90 day chunks, should be well below 16MB, 200k row API limits
            sfa_session.addForecastData(data_chunk, forecast_uuid)              #Add data chunk
            logging.info("\t" + str(round(idx/total_len*100,1)) + "% (" + str(idx) + "/" + str(total_len) + ")")

    logging.info("Done updating data.")


class SFA():

    def __init__(self, config):
        self.max_tries = config["maxTries"]
        self.upload_error_count = {}            #Tracks upload errors per observation
        try:
            sfa_tok = config["cachedToken"]
            self.session = sfa_api.APISession(sfa_tok)
            self.session.get_user_info()        #Make sure token is still valid
            logging.info("Using cached token")
        except (KeyError, Exception):
            #Get a new token
            logging.info("Getting new token")
            sfa_tok = sfa_api.request_cli_access_token(config["user"],config["password"])
            self.session = sfa_api.APISession(sfa_tok)
            logging.info("SFA Token: " + sfa_tok)
        self.cachedToken = sfa_tok

    def getSites(self):
        sfa_site_list = self.session.list_sites()
        s_count = 0
        p_count = 0
        for s in sfa_site_list:
            if type(s) is sfa_dm.Site:
                s_count += 1
            else:
                sfa_site_list.remove(s)
                p_count += 1
        logging.info("Sites: " + str(s_count) + "\tPower Plants (ignored): " + str(p_count))
        return sfa_site_list

    def addSite(self, s):
        sfa_site = sfa_dm.Site(name=s["name"],latitude=s["lat"],longitude=s["long"],elevation=s["alt"],timezone=s["timezone"])
        logging.info("Adding site: " + str(sfa_site))
        sfa_site = self.session.create_site(sfa_site)
        return sfa_site

    def listObservations(self):
        return self.session.list_observations()
    
    def getObservationTimeRange(self, obs_uuid):
        return self.session.get_observation_time_range(obs_uuid)
    
    def addObservation(self, obs):
        sfa_obs = sfa_dm.Observation(name=obs["obs_name"], variable=obs["measurement"].lower(), interval_value_type='interval_mean', interval_length=pd.Timedelta('0 days 00:01:00'), interval_label='ending', site=obs["site"], uncertainty=None, extra_parameters=str({"instrument": obs["instrument"]}))
        logging.info("Adding observation: " + str(sfa_obs))
        sfa_obs = self.session.create_observation(sfa_obs)
        return sfa_obs
        
    def addObsData(self, df, obs_uuid):
        temp = df.copy()    #Seems like there should be a more effecient way to do this, but since df is already a slice...
        temp["quality_flag"].fillna(1, inplace=True)
        temp["quality_flag"] = temp["quality_flag"].astype(int)
        try:
            self.session.post_observation_values(obs_uuid, temp)
            logging.info("\tUploaded " + str(len(temp)) + " rows")
            self.upload_error_count.update({obs_uuid: 0})             #Clear error count
        except Exception as e:
            logging.error("Error upoading: " + str(e))
            try:
                if self.upload_error_count[obs_uuid] >= self.max_tries:
                    raise Exception("Max tries exceeded: " + str(e))
                self.upload_error_count[obs_uuid] += 1
            except KeyError:
                self.upload_error_count[obs_uuid] = 1
        return
    
    def listForecasts(self):
        return self.session.list_forecasts()
        
    def getForecastTimeRange(self, forecast_uuid):
        return self.session.get_forecast_time_range(forecast_uuid)

    def addForecast(self, forecast):
        sfa_obs = sfa_dm.Forecast(name=forecast["forecast_name"], variable=forecast["variable"].lower(), issue_time_of_day=forecast["issue_start_datetime"].round('1min').strftime("%H:%M"),
                                  lead_time_to_start=pd.Timedelta(minutes=forecast["lead_time"]), run_length=pd.Timedelta(minutes=1),
                                  interval_value_type='instantaneous', interval_length=pd.Timedelta('0 days 00:01:00'), interval_label='instant', site=forecast["site"])
        logging.info("Adding forecast: " + str(sfa_obs))
        sfa_obs = self.session.create_forecast(sfa_obs)
        return sfa_obs
        
    def addForecastData(self, df, forecast_uuid):
        #temp = df.copy()    #Seems like there should be a more effecient way to do this, but since df is already a slice...
        try:
            self.session.post_forecast_values(forecast_uuid, df["value"])
            logging.info("\tUploaded " + str(len(df["value"])) + " rows")
            self.upload_error_count.update({forecast_uuid: 0})             #Clear error count
        except Exception as e:
            logging.error("Error uploading: " + str(e))
            try:
                if self.upload_error_count[forecast_uuid] >= self.max_tries:
                    raise Exception("Max tries exceeded: " + str(e))
                self.upload_error_count[forecast_uuid] += 1
            except KeyError:
                self.upload_error_count[forecast_uuid] = 1
        return

if __name__ == "__main__":
    #syncObs()
    syncForecasts()