from config_handler import handle_config
from datetime import datetime,timezone
import glob
import logging
import multiprocessing
from os import system, path, chmod, rename
import pysolar.solar as ps
import socket
import ssl
import sys
from threading import Event, Thread
import time
import traceback
import urllib.request as req

try:
    from os import mkdirs  # for python3.5
except:
    from os import makedirs as mkdirs # for python3.6 and above

socket.setdefaulttimeout(2);

# util function to call a routine at a specified interval
def call_repeatedly(intv, func, *args):
    stopped = Event()
    def loop():
        i = 0
        while not stopped.wait(intv): # the first call is in `intv` secs
            # print("Invoking {} for the {}th time".format(func.__name__, i))
            func(*args)
            i += 1
    Thread(target=loop).start()    
    return stopped.set

# move files from cache to output directory
def flush_files(cams):
    for camera in cams:
        fns = glob.glob(cachepath+camera+'/*jpg')
        # print(camera,len(fns))
        for fn in fns:
            doy = fn[-18:-10]
            dest = "{}{}/{}".format( imagepath, camera, doy )
            if not path.isdir(dest):
                if SAFE_MODE:
                    print("mkdirs " + dest)
                else:
                    mkdirs(dest)
                    chmod(dest, 0o755)
            if SAFE_MODE:
                print("rename {} to {}/{}_{}".format(fn,dest,camera,fn[-18:]))
            else:
                rename( fn, "{}/{}_{}".format( dest, camera, fn[-18:] ) )

# download images from cameras to "cache" 
# and also make a copy to "latest" directory
# the "latest" directory enables the web dashboard to show real time images
def makeRequest(cam):
    starttime = datetime.utcnow()
    timestamp = starttime.strftime( "%Y%m%d%H%M%S" )

    # for improper ssl certificates, try this to ignore CERTs
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    proxy = req.ProxyHandler({})
    opener = req.build_opener(proxy,req.HTTPSHandler(context=context))
    req.install_opener(opener)

    fn = cachepath + cam + "/{}_{}.jpg".format( cam, timestamp )
    if SAFE_MODE:
        print( "Would retrieve {} to {}".format( urls[cam]+url_suffix, fn ) )
    else:
        # print("Attempting retrieve {} to {}".format(urls[cam]+url_suffix, fn))
        req.urlretrieve( urls[cam] + url_suffix, fn )
        chmod(fn,0o755); # set the permission

    fn_latest = latest + cam + '_latest.jpg'
    if SAFE_MODE:
        print( "Would copy {} to {}".format( fn, fn_latest ) )
    else:
        system( "cp {} {}".format( fn, fn_latest ) )
        chmod( fn_latest, 0o755 ); # set the permission

if __name__ == "__main__":
    # load the configuration file
    cp = handle_config( 
      metadata={"invoking_script":"image_downloader"}, header="downloader"
    )
    site = cp["site_id"]
    config = cp['downloader']

    SAFE_MODE = config['safe_mode'] # run without consequences?
    if SAFE_MODE:
        print( "Initializing image_downloader in safe_mode" )

    url_suffix = config['network']['url_suffix']

    flush_interval = config["flush_interval"]
    interval_day = config['interval']['interval_day']
    interval_night = config['interval']['interval_night']

    site_config = config[site] # parameters that vary between sites
    lat = site_config['geolocation']['lat']
    lon = site_config['geolocation']['lon']

    site_paths = config['paths'][site]
    cachepath = site_paths['cache_path']
    latest = site_paths['latest_path']
    imagepath = site_paths['img_path']
    logpath = site_paths['logging_path']

    # create the directories used if they do not already exist
    for dest in [cachepath,latest,imagepath]:
        if not path.isdir(dest) and not SAFE_MODE:
            mkdirs(dest)
            chmod(dest,0o755)

    urls = {}
    for cameraID, url in cp['cameras'][site]['urls'].items():
        cameraID = cameraID.upper()
        urls[cameraID] = url

        dest = cachepath + cameraID
        if not path.isdir(dest) and not SAFE_MODE:
            mkdirs(dest)
            chmod(dest,0o755)

        desti = imagepath + cameraID
        if not path.isdir(dest) and not SAFE_MODE:
            mkdirs(dest)
            chmod(dest,0o755)
    
    # initialize the logger
    logging.basicConfig(format='%(asctime)s [%(funcName)s] [%(process)d %(thread)d] %(levelname)s: %(message)s',\
                        level=logging.INFO,filename=path.join(logpath,'image_downloader.log'),filemode='w')
    logger=logging.getLogger(__name__)

    ### Start loops
    # invoke flush_files every flush_interval seconds
    flush_event = call_repeatedly( flush_interval, flush_files, urls )

    p = multiprocessing.Pool( len(urls) )
    while (True):
        try:
            day_flag = ps.get_altitude( lat, lon, datetime.now(timezone.utc) ) > 5

            # invoke makeRequest once per camera every intv seconds
            intv = interval_day if day_flag else interval_night
            saveimage_event = call_repeatedly(intv, p.map_async, makeRequest, urls)
            
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
            # stop save_image from looping so we can restart it with the new intv
            try:
                saveimage_event()
            except:
                pass
