import glob, os, sys
from configparser import ConfigParser
from ast import literal_eval
from pytz import timezone
import image_pipeline
import stitch_pipeline
import features_pipeline
import camera
#import multiprocessing
import ray
import numpy as np
import ephem
import utils
from datetime import datetime, timedelta

#############################################################################

REPROCESS = False  # Reprocess already processed file?
INTERVAL = 0.5  # 0.5 min or 30 sec
use_min_imgs = False
chunk_size = 100        #cores_to_use
stitch_chunk_size = int(chunk_size/2) #int(chunk_size / len(cfg.cid_flat))
attempt_restore = True  #Attempt to restore partial netCDF files from tmp directory

do_preprocessing = True
do_height_stitching = True
do_features = True

#############################################################################

def catch(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return None

class Config(object):        
        
    def __init__(self):
       # Load the information from the configuration file
        try:
            # The module argparse should be use to handle the command-line interface
            try:
                config_file = sys.argv[1]
            except Exception:
                config_file = "./config.conf"

            # Read the configuration file
            print("Reading the configuration file " + config_file)
            cp = ConfigParser()
            cp.read(config_file)

            # The following variables are as defined in config.conf
            # List of camera IDs
            self.all_cameras = literal_eval(cp["cameras"]["all_cameras"])
            # The cameras are grouped in groupes to determine height
            self.height_group = literal_eval(cp["cameras"]["height_group"])
            # Pairs of cameras (stitch criteria)
            self.stitch_pair = literal_eval(cp["cameras"]["stitch_pair"])
            # IDs
            self.cameras_id = literal_eval(cp["cameras"]["cameras_id"])
            # Location of the cameras (lat/lon)
            self.location = literal_eval(cp["cameras"]["location"])
            # Parameters
            self.param = literal_eval(cp["cameras"]["parameters"])

            # List of cameras IDs + stitch pair (without repeated elements)
            self.cid_flat = list(set(self.cameras_id + [self.stitch_pair[camID] for camID in self.cameras_id]))
            print("Using cams: " + str(self.cid_flat))
            
            # Forecast params
            try:
                start_date = datetime.strptime(literal_eval(cp["forecast"]["start_date"]), "%Y%m%d")
                end_date = datetime.strptime(literal_eval(cp["forecast"]["end_date"]), "%Y%m%d")
                self.days = [y.strftime('%Y%m%d') for y in [start_date+timedelta(days=x) for x in range((end_date-start_date).days + 1)]]
                
            except Exception:
                self.days = literal_eval(cp["forecast"]["days"])
                
            self.lead_minutes = literal_eval(cp["forecast"]["lead_minutes"])
            self.lead_steps = [lt / INTERVAL for lt in self.lead_minutes]

            # Paths
            self.inpath = literal_eval(cp["paths"]["inpath"])
            self.tmpfs = literal_eval(cp["paths"]["tmpfs"])
            try:
                self.tmpfs_in = literal_eval(cp["paths"]["tmpfs_in"])
            except KeyError:
                self.tmpfs_in = self.tmpfs
            self.stitch_path = literal_eval(cp["paths"]["stitch_path"])
            self.static_mask_path = literal_eval(cp["paths"]["static_mask_path"])
            self.outpath = literal_eval(cp["paths"]["feature_path"])
            
            # GHI params
            self.GHI_Coor = literal_eval(cp["GHI_sensors"]["GHI_Coor"])
            self.GHI_loc = [self.GHI_Coor[key] for key in sorted(self.GHI_Coor)]
            self.GHI_loc = np.array(self.GHI_loc)

            # Define time zone (EST)
            try:
                self.cam_tz = timezone(cp["cameras"]["timezone"])
                print("Using camera timezone: %s" % str(self.cam_tz))
            except Exception:
                self.cam_tz = timezone("utc")
                print("Error processing camera timezone config, assuming UTC")

            # Processing params
            self.chunk_size = int(cp["server"]["preprocess_chunk_size"])
            self.stitch_chunk_size = int(cp["server"]["stitch_chunk_size"])
            self.feature_chunk_size = int(cp["server"]["feature_chunk_size"])
            
            # Define number of cores
            #try:
            #    self.cores_to_use = int(cp["server"]["cores_to_use"])
            #except Exception:
            #    self.cores_to_use = 20
            self.cp = cp
           
        except KeyError as e:
            print("Error loading config: %s" % e)
            exit()
        
        
def init():
    global cameras, rayNone, noneImg, forecast_stats_total

    ray.init(address='auto', ignore_reinit_error=True)
    #ray.init(local_mode=True)
    #ray.init()
    rayNone = ray.put(None)                             #None object for keeping types happy in lists
    noneImg = ray.put(image_pipeline.ImageShare())      #create empty image object as placeholder


    print("DAYS: %s" % cfg.days)
    forecast_stats_total = np.zeros((len(cfg.GHI_loc), len(cfg.lead_minutes)))

    # Allow interactive plots during debugging
    # plt.ioff()  #Turn off interactive plotting for running automatically

    # Initialize the list of camera objects (only once)
    cameras = {};

    if not os.path.isdir(cfg.static_mask_path):
        try:
            print("Creating directory " + cfg.static_mask_path)
            os.mkdir(cfg.static_mask_path)
        except OSError:
            print("Cannot create directory " + cfg.static_mask_path)
            exit()

    for camID in cfg.all_cameras:

        lat = cfg.location[camID][0]
        lon = cfg.location[camID][1]

        nx0 = ny0 = cfg.param[camID][0]
        xstart = cfg.param[camID][2]
        ystart = cfg.param[camID][1]
        rotation = cfg.param[camID][3]
        beta = cfg.param[camID][4]
        azm = cfg.param[camID][5]
        c1 = cfg.param[camID][6]
        c2 = cfg.param[camID][7]
        c3 = cfg.param[camID][8]

        cameras[camID] = ray.put(camera.camera(camID, lat, lon, nx0, ny0, \
                                       xstart, ystart, rotation, \
                                       beta, azm, c1, c2, c3, \
                                       max_theta=70, \
                                       nxy=1000, \
                                       timezone=cfg.cam_tz, \
                                       fpath=cfg.static_mask_path))
                                       

    #############################################################################

        
def preprocess(day, cfg):
    # Create output dir
    dir = cfg.tmpfs + day[:8]
    if not os.path.isdir(dir):
        try:
            print("Creating directory "+dir)
            os.mkdir(dir)
        except OSError:
            print("Cannot create directory "+dir)
            return False

    print("*** Processing %s ***" % day)
    ymd = day[:8]
    
    
    existing_stitches = get_existing_stitches(day, cfg)
    if len(existing_stitches) > 0:
        print("Found " + str(len(existing_stitches)) + " existing stitches")
    
    # Create all the images
    for cid in cfg.cid_flat:

        camera_cur = cameras[cid]

        # List of files
        print('\n\nPreprocessing ' + cfg.inpath + cid + '/' + ymd + '/')
        img_files_all = sorted(glob.glob(cfg.inpath + cid + '/' + ymd + '/' + cid + '_' + day + '*jpg'))
        print("  Found %i image files for %s" % (len(img_files_all), cid))
        
        img_files_all = [i for i in img_files_all if i[-18:-4] not in existing_stitches]
        print("  After skipping existing stitches: %i" % len(img_files_all))
        
        night_list = check_solar_angle(img_files_all, cid, cfg)
        img_files_all = [i for i in img_files_all if i not in night_list]
        print("  After skipping night images: %i" % len(img_files_all))        
        
        if len(img_files_all) <= 0:
            continue
            
        min_prev_obs_last = None   #need to be defined here in order to pass last image to the next chunk
        
        for f_idx in range(0, len(img_files_all), cfg.chunk_size):
            print("  Processing " + str(f_idx) + "-" + str(f_idx+cfg.chunk_size) + " of " + str(len(img_files_all)))
            img_files = [f for f in img_files_all[f_idx:f_idx+cfg.chunk_size]]
            img_obs = [image_pipeline.Image.remote(f, camera_cur) for f in img_files]
            
            if attempt_restore:
                print("    Checking existing data...")
                nc_files = [image_pipeline.get_ncdf_curr_filename(f,cfg.tmpfs_in) for f in img_files]
                existing_data = ray.get([i.restore_ncdf.remote(filename=f, ignore_missing=True) for f, i in zip(nc_files, img_obs)])   #attempt to restore exisitng data
                if sum([len(i) for i in existing_data]) > 0:
                    print("    Found:\n\tnetCDFs: " + str(sum([i.count("img") for i in existing_data])) + \
                            "\n\tUndistorted images: " + str(sum([i.count("rgbu") for i in existing_data])) + \
                            "\n\tBright masks: " + str(sum([i.count("bright_mask") for i in existing_data])) + \
                            "\n\tCloud masks: " + str(sum([i.count("cloud_mask") for i in existing_data])) + \
                            "\n\tCloud motion: " + str(sum([i.count("cloud_motion") for i in existing_data]))) # + \
                            #"\n\tCloud heights: " + str(sum([i.count("cloud_base_height") for i in existing_data])))               
                    if all([d in i for i in existing_data for d in ["img","rgbu","bright_mask","cloud_mask","cloud_motion"]]):
                        print("\tSet already complete, skipping")
                        min_prev_obs_last = img_obs[-1].get_image_min.remote(include_rgbu=True) # cache last image in case next set needs it
                        continue
            else:
                existing_data = [[None]] * len(img_obs)
            
            img_files = ray.get([i.read_image.remote() if "img" not in existing_data[idx] or len(existing_data[idx]) <=1 else ray.put(img_files[idx]) for idx, i in enumerate(img_obs)])
            print("    Loaded " + str(sum(x is not None for x in img_files)) + " images\n\tPreprocessing...")          
            
            results = [i.undistort_image.remote() if "rgbu" not in existing_data[idx] else ray.put(True) for idx, i in enumerate(img_obs)]
            print("\t  Preprocessed: " + str(sum(x is not None for x in results)) + "\n\tAdding to memory...")
            
            min_obs_masks = ray.get([i.get_image_min.remote(include_rgbu=True) for i in img_obs])  #create reducded memory img objects
            print("\t  Added min imgs to dict: " + str(sum(x is not None for x in min_obs_masks)))
            
            prev_files = [image_pipeline.get_prev_filename(f) for f in img_files]
            min_prev_obs = [catch(lambda:min_obs_masks[catch(img_files.index, p)]) for p in prev_files]
            min_prev_obs[0] = min_prev_obs_last                                  #load first element from previous loop
            min_prev_obs_last = min_prev_obs[-1]                                 #save last element for next loop
            min_prev_obs = [noneImg if p is None else p for p in min_prev_obs]   #replace Nones with placeholders
            print("\t  Prepared prev images: " + str(sum(x is not None for x in min_prev_obs)) + "\n\tProcessing bright mask...")
            
            results = [i.compute_bright_mask.remote(min_prev_obs[idx]) if "bright_mask" not in existing_data[idx] else ray.put(True) for idx, i in enumerate(img_obs)]
            print("\t  Bright mask: " + str(sum(x is not None for x in results)) + "\n\tProcessing cloud mask...")
            
            results = [i.compute_cloud_mask.remote(min_prev_obs[idx]) if "cloud_mask" not in existing_data[idx] else ray.put(True) for idx, i in enumerate(img_obs)]
            print("\t  Cloud mask: " + str(sum(x is not None for x in results)) + "\n\tProcessing cloud motion...")
            
            results = [i.compute_cloud_motion.remote(min_prev_obs[idx]) if "cloud_motion" not in existing_data[idx] else ray.put(True) for idx, i in enumerate(img_obs)]
            print("\t  Cloud motion: " + str(sum(x is not None for x in results)) + "\n\tSaving netCDFs...")
            
            nc_files = [image_pipeline.get_ncdf_curr_filename(f,cfg.tmpfs) for f in img_files]         #Update nc file list (though it should be the same)
            results = ray.get([i.save_netcdf.remote(f, inmemory=False) for f, i in zip(nc_files, img_obs)])
            print("\t  Saved netCDFs: " + str(sum(x is not None for x in results)))
            
            #min_obs[cid].update({f: i.get_image_min.remote() for f, i in zip(nc_files, img_obs)})  #create reducded memory img objects
            #print("\t  Objects in min imgs for " + cid + ": " + str(len(min_obs[cid])))
    return True


def stitch(day,cfg):
    print("\n\nStitching " + str(day))
    # Create file index by date and cam
    img_dict = {}
    if sorted(cfg.cid_flat) != sorted(cfg.all_cameras):
        print("Warning: cfg.cid_flat != all_cameras, check config to make sure this is intended")   
        print("cfg.cid_flat:    " + str(cfg.cid_flat) + "\nall_cameras: " + str(cfg.all_cameras))
        
    existing_stitches = get_existing_stitches(day, cfg)
    if len(existing_stitches) > 0:
        print("Found " + str(len(existing_stitches)) + " existing stitches")    
    
    for cid in cfg.cid_flat:
        #if cid not in cfg.cid_flat or stitch_pair[cid] in selected:
        #    continue;
        ncdfs = sorted(glob.glob(cfg.tmpfs + day[:8] + '/' + cid + '_' + day + '*.nc'));
        print("%s: %i netcdf files found" % (cid, len(ncdfs)))
        ncdfs = [i for i in ncdfs if i[-17:-3] not in existing_stitches]
        print("  After skipping existing stitches: %i" % len(ncdfs))    
        for ncdf in ncdfs:
            try:
                img_dict[ncdf[-17:-3]].update({cid: ncdf})
            except KeyError:
                img_dict[ncdf[-17:-3]] = {cid: ncdf}
    
    for f_idx in range(0, len(img_dict), cfg.stitch_chunk_size):
        print("  Processing " + str(f_idx) + "-" + str(f_idx+cfg.stitch_chunk_size) + " of " + str(len(img_dict)))
    
        img_times = [f for f in list(img_dict.keys())[f_idx:f_idx+cfg.stitch_chunk_size]]
        min_imgs = {}
        if True:
            img_obs = {t:{(cam, img_dict[t][cam]): image_pipeline.Image.remote(img_dict[t][cam], cameras[cam]) for cam in img_dict[t]} for t in img_times}
            results = [i.restore_ncdf.remote() for t in img_obs.values() for k, i in t.items()]             # restore netCDF data
            min_imgs = {t:{k: i.get_image_min.remote(include_rgbu=True) for k, i in o.items()} for t, o in img_obs.items()}          # create reduced memory img objects
            print("\tLoaded netCDFs: " + str(sum(x is not None for x in results)))
            del img_obs
            
        stitch_obs = {t:stitch_pipeline.Stitch.remote(t, min_imgs[t], cfg.stitch_path, cfg.height_group) for t in img_times}                      # create stitch objects
        
        #Calc heights
        results = {t: s.process_cloud_heights.remote() for t, s in stitch_obs.items()}
        print("\tCalculated heights: " + str(sum(x is not None for x in results.values()))) #str({t: ray.get(r) for t, r in results.items()}).replace(',',',\n\t  '))
        
        #Create stitched image
        results = ray.get([s.generate_stitch.remote() for t, s in stitch_obs.items()])
        print("\tCreated stitched images: " + str(sum(x is not None for x in results)))
       

        
def get_existing_stitches(day, cfg):
    flist = sorted(glob.glob(cfg.stitch_path + day[:8] + '/' + day + '*.nc'))
    try:
        existing_times = [f[-17:-3] for f in flist]
    except KeyError:
        return None
    except TypeError:
        return None
    return existing_times

def check_solar_angle(flist, camID, cfg, max_saa=75):
    lat = cfg.location[camID][0]
    lon = cfg.location[camID][1]
    obs = ephem.Observer();
    obs.lat = str(lat)
    obs.lon = str(lon)    
    sun = ephem.Sun();    
    
    night_list = []
    for f in flist:
        t_std = utils.localToUTC(datetime.strptime(f[-18:-4], '%Y%m%d%H%M%S'), cfg.cam_tz)
        obs.date = t_std.strftime('%Y/%m/%d %H:%M:%S')
        sun.compute(obs);
        saa = np.pi / 2 - sun.alt;
        if saa > np.deg2rad(75):
            night_list += [f]
        
    return night_list
    

if __name__ == "__main__":

    cfg = Config()
    
    init()

    for day in cfg.days:

        if not preprocess(day, cfg):
            continue    # if there's a critical error, skip to the next day

        stitch(day, cfg)
        
        forecast_stats = features_pipeline.extract(day, cfg)
        if forecast_stats is not False:
            forecast_stats_total += forecast_stats
    
    #Save stats if they exist
    if sum(forecast_stats_total.flat) > 0:
        np.savetxt(cfg.outpath + 'forecast_stats_run_' + day[:8] + '.csv', forecast_stats_total, fmt="%i", delimiter=',')        
    print("*** Done with " + day + "***")

print("*** Done with full set ***")
