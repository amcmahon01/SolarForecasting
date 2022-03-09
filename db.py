from datetime import timedelta
import logging
import pandas as pd
from sqlalchemy import create_engine
import pymysql


class DBengine():

    def __init__(self, config):
        logging.info("Connecting to " + config["host"] + "/"+ config["database"])
        self.engine = create_engine("mysql+pymysql://" + config["user"] + ":" + config["password"] + "@" + config["host"] + "/"+ config["database"] + "?charset=utf8mb4&autocommit=true&local_infile=1")

    def getSites(self):
        try:
            sites = pd.read_sql_table('sites', self.engine)
            site_d = sites.to_dict('records')
            return site_d
        except Exception as e:
            logging.error("Error getting sites: " + str(e))
            raise e

    def getSensors(self):
        sensors = pd.read_sql('SELECT sensors.sensor_id, sensors.site_id, sensors.measurement, sensors.instrument, sensors.units, sites.name AS site_name FROM sensors INNER JOIN sites ON sensors.site_id = sites.site_id', self.engine)
        sensors_d = sensors.to_dict('records')
        return sensors_d

    def getObsData(self, sensor):
        logging.info("Getting data for " + sensor["obs_name"])
        try:    #Check for exclusion range  ****FUTURE: Add "value gaps" check
            start_dt = sensor["start_dt"].tz_convert("EST").strftime("%Y-%m-%d %H:%M:%S")
            end_dt = sensor["end_dt"].tz_convert("EST").strftime("%Y-%m-%d %H:%M:%S")
            q_dt = " AND NOT (TIMESTAMP BETWEEN '" + start_dt + "' AND '" + end_dt + "')"
        except (KeyError, ValueError):
            q_dt = ""
        data = pd.read_sql('SELECT TIMESTAMP, value, quality_flag FROM observations_1min_avg WHERE sensor_id=' + str(sensor["sensor_id"]) + q_dt, self.engine)
        data.set_index(data["TIMESTAMP"], inplace=True)
        data.drop(columns="TIMESTAMP", inplace=True)
        data = data.tz_localize("EST")
        logging.info("\tRecords found: " + str(len(data)))
        return data

    def getForecasts(self):
        sensors = pd.read_sql('SELECT forecast_configs.config_id, forecast_configs.name, forecast_configs.sensor_id, forecast_configs.issue_start_datetime, forecast_configs.lead_time, forecast_configs.start_datetime, forecast_configs.end_datetime, forecast_configs.variable, sensors.site_id FROM forecast_configs INNER JOIN sensors ON sensors.sensor_id = forecast_configs.sensor_id', self.engine)
        sensors_d = sensors.to_dict('records')
        return sensors_d

    def getForecastData(self, forecast):
        logging.info("Getting data for " + forecast["forecast_name"])
        try:    #Check for exclusion range  ****FUTURE: Add "value gaps" check
            start_dt = forecast["start_dt"].tz_convert("EST").strftime("%Y-%m-%d %H:%M:%S")
            end_dt = forecast["end_dt"].tz_convert("EST").strftime("%Y-%m-%d %H:%M:%S")
            q_dt = " AND NOT (TIMESTAMP BETWEEN '" + start_dt + "' AND '" + end_dt + "')"
        except (KeyError, ValueError):
            q_dt = ""
        data = pd.read_sql('SELECT TIMESTAMP, value FROM forecasts WHERE forecast_config=' + str(forecast["config_id"]) + q_dt + " ORDER BY TIMESTAMP ASC", self.engine)
        data.set_index(data["TIMESTAMP"], inplace=True)
        data.drop(columns="TIMESTAMP", inplace=True)
        data = data.tz_localize("EST")
        logging.info("\tRecords found: " + str(len(data)))
        return data

    def updateSite(self, site):
        logging.info("Updating metadata for " + site.name)
        query = 'UPDATE sites SET sfa_site_id="' + site.site_id + '", climate_zones="' + str(site.climate_zones) + '" WHERE sites.name="' + site.name + '"'
        result = self.engine.execute(query)
        return result

    def updateObservation(self, obs):
        logging.info("Updating metadata for " + obs.name)
        query = 'UPDATE sensors SET sfa_obs_id="' + obs.observation_id + '" WHERE sensors.name="' + obs.name + '"'
        result = self.engine.execute(query)
        return result
    
    def updateForecast(self, forecast):
        logging.info("Updating metadata for " + forecast.name)
        query = 'UPDATE forecast_configs SET sfa_obs_id="' + forecast.observation_id + '" WHERE sensors.name="' + forecast.name + '"'
        result = self.engine.execute(query)
        return result

    def loadObsData(self, fname, sensor_id, pad_fields=3, ignore_lines=4):
        #create forecast
        fname = fname.replace("\\", "/") #make sql happy

        # Note: ESCAPED BY '' is used instead of ESCAPED BY '"' because of an ancient bug (https://bugs.mysql.com/bug.php?id=39247)
        q = "LOAD DATA CONCURRENT LOCAL INFILE '" + fname \
            + "' IGNORE INTO TABLE `observations_1sec` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY \'\"\' ESCAPED BY \'\' LINES TERMINATED BY '\n' IGNORE " \
            + str(ignore_lines) + " LINES (`TIMESTAMP`," + "@dummy,"*pad_fields + "`value`) SET sensor_id = " + str(sensor_id)

        result = self.engine.execute(q)
        logging.info("\tLoaded " + str(result.rowcount) + " rows.")

        return result

    def createForecast(self, fname):
        logging.info("Creating forecast for " + fname)
        
        df = pd.read_csv(fname)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
        df["forecast_time"] = pd.to_datetime(df["forecast_time"], unit='s')
        issue_start_datetime = df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
        start_datetime = df["forecast_time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
        end_datetime = df["forecast_time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        #forecast_len = int((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()/60)
        lead_time = df["lead_minutes"].iloc[0]
        sensor_id = df["location"].iloc[0] + 1
        forecast_name = fname.split("forecast_")[-1].split("_" + str(lead_time) + "_total.csv")[0]

        fields = [forecast_name, str(sensor_id), issue_start_datetime, str(lead_time), start_datetime, end_datetime, 'ghi']
        logging.debug(fields)

        query = 'INSERT INTO forecast_configs (name, sensor_id, issue_start_datetime, lead_time, start_datetime, end_datetime, variable) VALUES ("' \
                + '","'.join(fields) + '")'

        result = self.engine.execute(query)
        logging.info(result)

        config_id_query = "SELECT config_id FROM forecast_configs ORDER BY config_id DESC LIMIT 1"
        config_id = self.engine.execute(config_id_query).fetchall()[0][0]
        logging.info("config_id: " + str(config_id))

        return config_id


    def loadForecastData(self, fname, forecast_config):
        #create forecast
        fname = fname.replace("\\", "/") #make sql happy

        fields = "(@ts,@ft,@dummy,@dummy,`value`) SET forecast_config=" + str(forecast_config) +", TIMESTAMP = FROM_UNIXTIME(@ts), forecast_time = FROM_UNIXTIME(@ft);"

        # Note: ESCAPED BY '' is used instead of ESCAPED BY '"' because of an ancient bug (https://bugs.mysql.com/bug.php?id=39247)
        q = "LOAD DATA CONCURRENT LOCAL INFILE '" + fname + "' IGNORE INTO TABLE `forecasts` \
            FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY \'\"\' ESCAPED BY \'\' LINES TERMINATED BY '\n' IGNORE 1 LINES " + fields

        result = self.engine.execute(q)
        logging.info("\tLoaded " + str(result.rowcount) + " rows.")

        return result