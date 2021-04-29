# download GHI values from pyranometers through HTTP
# dev. and tested in anaconda py35 environment
# pass yaml config file on commandline, see config_handler for details.
# derived from image_downloader.py
# 
from config_handler import handle_config
from datetime import datetime,timezone
import pandas as pd
import logging,logging.config
import multiprocessing
from os import system, chmod, rename
import pysolar.solar as ps
import socket
import sys
import math
from threading import Event, Thread
import time
import traceback
# note: pathlibs Path object have their own open method, use that for python <3.6
from pathlib import Path

# not sure what the default socket time-out means, but increased it to 6s
# for GHI downloads (was 2s for cameras).
socket.setdefaulttimeout(6);

# util function to call a routine at a specified interval
def call_repeatedly(intv, func, *args):
    stopped = Event()
    def loop():
        i = 0
        while not stopped.wait(intv):
            func(*args)
            i += 1
    Thread(target=loop).start()    
    return stopped.set

# move files from cache to output directory
def flush_files(cams):
    for camera in cams:
        source=Path(cachepath,camera)
        fns = source.glob('*.csv')

        for f in fns:
            # fn is string of full path
            fn=str(f)
            # fname is string of last path component, i.e. filename.
            fname=f.name
            doy = fname[-18:-10]
            dest = Path( imagepath, camera, doy )
            if not dest.is_dir():
                if SAFE_MODE:
                    logger.info("mkdirs " + dest)
                else:
                    dest.mkdir(exist_ok=True)
            if SAFE_MODE:
                logger.info("rename {} to {}/{}".format(fn,dest,fname))
            else:
                rename( fn, "{}/{}".format( dest, fname ) )

# download interval_day sec of data from GHI sensor
# and also make a copy to "latest" directory
# the "latest" directory enables the web dashboard to show real time images
# maybe better not to reuse the image_downloader's "latest" and "cache" dirs
def makeRequest(GHIsensor):
    starttime = datetime.utcnow()
    doy=starttime.strftime( "%Y%m%d" )
    timestamp = doy+starttime.strftime( "%H%M%S" )
    dest=Path(GHIpath,GHIsensor,doy)
    dest.mkdir(parents=True,exist_ok=True)
    f = Path(dest,"{}_{}.csv".format( GHIsensor, timestamp ))
    fn=str(f)
    fname=f.name
    # need to add at least 1s to intv because we are missing 1 s between files
    # maybe better to go back to previous file, or adjust wait time to always
    # download at the same time e.g. :30 and :00
    # For now just adding 2% to take care of slow download speed of GHI
    url=GHIsensors[GHIsensor]['url']+GHI_suffix+str(math.ceil(1.02*intv))
    if SAFE_MODE:
        logger.info( "Would retrieve {} to {}".format( url, fn ) )
    else:
        try:
            dfs=pd.read_html(url,index_col=1,header=0)
        except:
            logger.error("{} could not be retrieved".format(url))
            return
        df=dfs[0]
        try:
            df.to_csv(fn)
        except:
            logger.error("{} could not be written".format(fn))

if __name__ == "__main__":
    cp = handle_config( 
      metadata={"invoking_script":"GHI_downloader"}, header="downloader"
    )
    site = cp["site_id"]
    config = cp['downloader']

    SAFE_MODE = config['safe_mode'] # run without consequences?
    if SAFE_MODE:
        print( "Initializing GHI_downloader in safe_mode" )

    GHI_suffix = config['GHI_suffix']

    flush_interval = config["flush_interval"]
    interval_day = config['interval_day']
    interval_night = config['interval_night']
# using site-specific config files on command line
    lat = config['geolocation']['lat']
    lon = config['geolocation']['lon']

    paths = cp['paths']
    cachepath = Path(paths['cache_path'])
    latest = Path(paths['latest_path'])
    imagepath = Path(paths['img_path'])
    logpath = Path(paths['logging_path'])
    logpath.mkdir(exist_ok=True)
    GHIpath = Path(paths['raw_GHI_path'])

    # create the directories used if they do not already exist
    if not GHIpath.is_dir() and not SAFE_MODE:
        GHIpath.mkdir(exist_ok=True)

    GHIsensors = {}
    for GHIsensor in cp['GHI_sensors'].keys():
        GHIsensor = GHIsensor.upper()
        GHIsensors[GHIsensor]=cp['GHI_sensors'][GHIsensor]

        dest = Path(GHIpath,GHIsensor)
        if not dest.is_dir() and not SAFE_MODE:
            dest.mkdir(exist_ok=True)

    # initialize the logger
    logfile=str(Path(logpath,'GHI_downloader.log'))
    logger=multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)
    fh=logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    formatter=logging.Formatter('%(asctime)s [%(name)s] [%(process)d %(thread)d] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

#    p = multiprocessing.Pool( len(GHIsensors) )
    while (True):
        try:
            day_flag = ps.get_altitude(lat, lon, datetime.now(timezone.utc)) > 5

            # invoke makeRequest once per camera every intv seconds
            intv = interval_day if day_flag else interval_night
            logger.debug("day_flag={}, intv={}".format(day_flag,intv))
#            saveimage_event = call_repeatedly(intv, p.map_async, makeRequest, GHIsensors)
            event_flags=list()
            for GHIsensor in GHIsensors.keys():
                saveimage_event = call_repeatedly(intv, makeRequest, GHIsensor)
                event_flags.append(saveimage_event)
            
            # check periodically if the sun has set or risen
            if day_flag:
                while ps.get_altitude( lat, lon, datetime.now(timezone.utc) ) > 5:
                    time.sleep(180)
            else:
                while ps.get_altitude( lat, lon, datetime.now(timezone.utc) ) <= 5:
                    time.sleep(600)

        except Exception as e:
            msg = traceback.trace_exc()
            logger.error( msg )
        finally:
            # end the save_image loop so we can restart it with the new intv
            try:
                for event_flag in event_flags:
                    event_flag()
            except:
                msg = traceback.trace_exc()
                logger.error( msg )
