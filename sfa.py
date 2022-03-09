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
            sfaConnection["maxTries"] = le(cp["sfaInfo"]["maxTries"])

            reports = {}
            reports["provider"] = le(cp["reports"]["provider"])
            reports["variables"] = le(cp["reports"]["variables"])
            reports["metrics"] = le(cp["reports"]["metrics"])
            reports["lead_times"] = le(cp["reports"]["lead_times"])

            forecast = {}
            forecast["overwrite_existing"] = le(cp["forecast"]["overwrite_existing"])
            forecast["sensors"] = le(cp["forecast"]["sensors"])
            forecast["config_name"] = cp["forecast"]["config_id"]

        except KeyError as e:
            logging.error("Missing config information: " + str(e))
            return
        
        return {"db": dbConnection, "sfa": sfaConnection, "reports": reports, "forecast":forecast}

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
    logging.info("Syncing observations...")
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

def syncForecasts(reports_only=False):
    logging.info("Syncing forecasts...")
    forecasts = db_engine.getForecasts()
    for forecast in forecasts:
        #Sync forecast metadata
        try:
            site = next(item for item in sites if item["site_id"] == forecast["site_id"])
        except KeyError:
            logging.warning("Site id " + str(forecast["site_id"]) + " not found, skipping.")
            continue
        if reports_only and (forecast["lead_time"] not in config["reports"]["lead_times"]):
            continue
        forecast["forecast_name"] = forecast["name"] + " " + site["name"] + " " + str(forecast["lead_time"]) + "min " + forecast["variable"]  #Site name + lead time + measurement should be used for all forecast names (ex: "LISF PB1 10min GHI")
        forecast["site"] = site_d[site["name"]]                              #Associate SFA site record
        if forecast["forecast_name"] not in forecasts_d.keys():           #Check which records are new    ****FUTURE: Should change to use UUID instead/in addition to****
            result = sfa_session.addForecast(forecast)                    #Add missing forecasts
            #db_engine.updateForecasts(result)                            ****FUTURE: Update local metadata
            forecasts_d.update({forecast["forecast_name"]:result})        #Add to local dict
        elif config["forecast"]["overwrite_existing"]:
            logging.info("Existing forecast found for %s, deleting (overwrite_existing=True)" % forecast["forecast_name"])
            old_uuid = forecasts_d[forecast["forecast_name"]].forecast_id
            result = sfa_session.deleteForecast(old_uuid)
            if result == 204:
                logging.info("\tSuccessfully deleted " + old_uuid)
            else:
                logging.warning("\tError removing existing forecast %s, skipping" % old_uuid)
                continue
            result = sfa_session.addForecast(forecast)                    #Add missing forecasts
            #db_engine.updateForecasts(result)                            ****FUTURE: Update local metadata
            forecasts_d.update({forecast["forecast_name"]:result})        #Add to local dict
        else:
            logging.info("Existing forecast found for %s, skipping (overwrite_existing=False)" % forecast["forecast_name"])
            continue

        #Sync forecast data
        forecast_uuid = forecasts_d[forecast["forecast_name"]].forecast_id                            #Get forecast UUID now that all should be in the local dict
        forecast["start_dt"], forecast["end_dt"] = sfa_session.getForecastTimeRange(forecast_uuid)    #Get existing data range (to be excluded from upload)
        data = db_engine.getForecastData(forecast)                         #Warning: Could be very large
        filled_data = fillMissing(data, method='nearest', useRounding=True)
        total_len = len(filled_data)
        logging.info("Uploading " + str(total_len) + " rows to " + forecast["forecast_name"] + " (" + forecast_uuid + ")")
        for idx, data_chunk in chunk(filled_data, 129600).items():         #Create 90 day chunks, should be well below 16MB, 200k row API limits
            sfa_session.addForecastData(data_chunk, forecast_uuid)              #Add data chunk
            logging.info("\t" + str(round(idx/total_len*100,1)) + "% (" + str(idx) + "/" + str(total_len) + ")")

    logging.info("Done updating data.")

def syncSites():
    logging.info("Syncing sites...")
    for s in sites:
        if s["name"] not in site_d.keys():                            #Check which records are new    ****FUTURE: Should change to use UUID instead/in addition to****
            site_result = sfa_session.addSite(s)                      #Add missing sites
            db_engine.updateSite(site_result)
            site_d.update({s["name"]:site_result})                    #Add to local site dict

def generatePersistence():
    logging.info("Generating persistence metadata... \
                \n\tUsing lead times: " + str(config["reports"]["lead_times"]) \
                + "\n\tUsing sensors: " + str(config["forecast"]["sensors"]) )
                
    for lead_time in config["reports"]["lead_times"]:
        for sensor_id in config["forecast"]["sensors"]:
            sensor = next(item for item in sensors if item["sensor_id"] == sensor_id)
            try:
                site = next(item for item in sites if item["site_id"] == sensor["site_id"])
            except KeyError:
                logging.warning("Site id " + str(sensor["site_id"]) + " not found, skipping.")
                continue
            forecast_name = "Persistence " + site["name"] + " " + str(lead_time) + "min " + sensor["measurement"]
            forecast_names = [f["name"] + " " + site["name"] + " " + str(lead_time) + "min " + sensor["measurement"] for f in forecasts if site["site_id"]==f["site_id"] and lead_time==f["lead_time"]]
            if forecast_name in forecast_names:
                logging.info("\t%s already exists, skipping." % forecast_name)
            else:
                db_engine.createPersistence(sensor, lead_time, site["name"])

def generateReports():
    logging.info("Generating reports...\n\tUpdating observation and forecast lists")
    sfa_obs = sfa_session.listObservations()
    sfa_forecasts = sfa_session.listForecasts()

    r_sensor_sites = {s["site_id"]:s for s in sensors if s["sensor_id"] in config["forecast"]["sensors"]}
    r_sites = {s["name"]:s for s in sites if s["site_id"] in r_sensor_sites.keys()}
    sfa_forecasts = [f for f in sfa_forecasts if f.provider==config["reports"]["provider"]
                                                 and int(f.lead_time_to_start.total_seconds()/60) in config["reports"]["lead_times"]
                                                 and f.site.name in r_sites.keys()]
    obs_d = {n.name: n for n in sfa_obs}
    forecasts_d = {n.name: n for n in sfa_forecasts}
    logging.info("Found %i observations and %i forecasts" % (len(sfa_obs), len(sfa_forecasts)))

    for lead_time in config["reports"]["lead_times"]:
        forecasts_grouped = {k:f for k,f in forecasts_d.items() if int(f.lead_time_to_start.total_seconds()/60)==lead_time}
        logging.info("  For %i min lead time, %i forecasts" % (lead_time, len(forecasts_grouped)))

        forecast_obs = []
        r_name = config["forecast"]["config_name"] + " " + str(lead_time) + "min"
        r_start = None
        r_end = None

        for f_name, sfa_for in forecasts_grouped.items():
            sfa_site = sfa_for.site
            obs_name = sfa_site.name + " " + sfa_for.variable.upper()
            f_name = sfa_for.name

            try:
                obs = obs_d[obs_name]
            except KeyError:
                logging.info("\t%s: No matching observation '%s' found, skipping" % (f_name, obs_name))
                continue

            f_start, f_end = sfa_session.getForecastTimeRange(sfa_for.forecast_id)
            forecast_ref = None
            try:
                r_start = min(r_start, f_start)
                r_end = max(r_end, f_end)
            except TypeError:
                r_start = f_start
                r_end = f_end
            try:
                if "persistence" not in f_name.lower():
                    ref_name = "Persistence " + sfa_site.name + " " + str(lead_time) + "min " + sfa_for.variable
                    forecast_ref = forecasts_d[ref_name]
            except KeyError:
                logging.warning("\tPersistence forecast not found, reference forecast will not be included")                
            forecast_obs += [sfa_dm.ForecastObservation(sfa_for, obs, forecast_ref)]
            logging.info("\t%s: Added to report" % f_name)

        logging.info("Generating report for %d forecasts from %s to %s" % (len(forecast_obs), r_start.strftime("%Y-%m-%d %H:%M:%S"), r_end.strftime("%Y-%m-%d %H:%M:%S")))
        result = sfa_session.generateReport(r_name, r_start, r_end, forecast_obs)
        try:
            logging.info("Report ID: %s\tStatus: %s" % (result.report_id, result.status))
        except Exception as e:
            logging.warning("\tError generating report, did not receive info from Arbiter")


class SFA():

    def __init__(self, config):
        self.config = config
        self.max_tries = config["maxTries"]
        self.upload_error_count = {}            #Tracks upload errors per observation
        self.connect()
    
    def connect(self):
        logging.info("Connecting to Solar Forecast Arbiter...")

        for attempt in range(0,self.max_tries):
            try:        
                try:
                    #sfa_tok = config["cachedToken"]
                    with open("sfa_token.txt", 'r') as f:
                        sfa_tok = f.readline()
                    self.session = sfa_api.APISession(sfa_tok)
                    self.session.get_user_info()        #Make sure token is still valid
                    logging.info("Using cached token")
                except (KeyError, Exception):
                    #Get a new token
                    logging.info("Getting new token")
                    sfa_tok = sfa_api.request_cli_access_token(self.config["user"],self.config["password"])
                    self.session = sfa_api.APISession(sfa_tok)
                    with open("sfa_token.txt", 'w') as f:
                        f.write(sfa_tok)
                        logging.info("\tCached for future requests")
                    logging.debug("SFA Token: " + sfa_tok)
                self.cachedToken = sfa_tok

                #verify connection
                self.session.get_user_info()
                return
            except Exception as e:
                logging.error("Error connecting, trying again in 5 seconds (%i/%i)" % (attempt, self.max_tries))
                time.sleep(5)
        logging.error("Max tries exceeded, cannot continue.")
        exit()

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
                                  interval_value_type='interval_mean', interval_length=pd.Timedelta('0 days 00:01:00'), interval_label='ending', site=forecast["site"])
        logging.info("Adding forecast: " + sfa_obs.name)
        sfa_obs = self.session.create_forecast(sfa_obs)
        return sfa_obs

    def deleteForecast(self, forecast_id):
        result = self.session.delete_forecast(forecast_id)
        return result
        
    def addForecastData(self, df, forecast_uuid):
        #temp = df.copy()    #Seems like there should be a more effecient way to do this, but since df is already a slice...
        for attempt in range(0,self.max_tries):      
            try:
                self.session.post_forecast_values(forecast_uuid, df["value"])
                logging.info("\tUploaded " + str(len(df["value"])) + " rows")
                self.upload_error_count.update({forecast_uuid: 0})             #Clear error count
                return
            except Exception as e:
                logging.error("Error uploading: " + str(e))
                
                if "UNAUTHORIZED" in str(e):
                    logging.info("Attempting to reconnect...")
                    self.connect()
                try:
                    if self.upload_error_count[forecast_uuid] >= int(self.max_tries):
                        break
                        #raise Exception("Max tries exceeded: " + str(e))
                    self.upload_error_count[forecast_uuid] += 1
                except KeyError:
                    self.upload_error_count[forecast_uuid] = 1
        logging.error("Max tries exceeded, skipping")

    def generateReport(self, r_name, r_start, r_end, forecast_obs):

        report_params = sfa_dm.ReportParameters(name=r_name, start=r_start, end=r_end, object_pairs=forecast_obs,
                 metrics=config["reports"]["metrics"], categories=('total', 'date'), forecast_fill_method='drop')

        report = sfa_dm.Report(report_params)
        result = self.session.create_report(report)

        return result
        


if __name__ == "__main__":

    startdt = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logFile = "Log_sfaSync_" + startdt + ".log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s', handlers=[logging.FileHandler(logFile), logging.StreamHandler()])

    config = readConfig()

    #Get nowcast metadata
    db_engine = DBengine(config["db"])
    sites = db_engine.getSites()
    sensors = db_engine.getSensors()
    forecasts = db_engine.getForecasts()

    #Get SFA metadata
    sfa_session = SFA(config["sfa"])
    sfa_sites = sfa_session.getSites()
    sfa_obs = sfa_session.listObservations()
    sfa_forecasts = sfa_session.listForecasts()

    site_d = {n.name: n for n in sfa_sites}
    obs_d = {n.name: n for n in sfa_obs}
    forecasts_d = {n.name: n for n in sfa_forecasts}

    syncSites()
    #syncObs()
    #generatePersistence()
    syncForecasts(reports_only=True)
    #generateReports()