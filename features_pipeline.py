import numpy as np
import os, sys, glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pickle
import multiprocessing
from multiprocessing import get_context
from datetime import datetime, timedelta
from configparser import ConfigParser
from ast import literal_eval
from pytz import timezone
from pvlib.location import Location
import pandas as pd
import stitch_pipeline
import utils
import ray
import warnings
import logging

#############################################################################

deg2km = 6367 * np.pi / 180
WIN_SIZE = 50  # half-width of bounding box integrated per GHI point
feature_chunk_size = 100
SAVE_FIG = True
REPROCESS = False  # Not yet implemented

#############################################################################

def chunks(lst, n):
    #https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield (i, lst[i:i + n])

#@ray.remote
def extract_MP(args):
    # TODO lead_steps should be input of the function

    iGHI, iy0, ix0, ny, nx, stch, timestamp, outpath, latlon, lead_minutes, lead_steps, cf_total = args

    if (iy0 < 0) or (ix0 < 0):
        logging.info("\tInvalid dimensions")
        return None

    loc = Location(latlon[0], latlon[1], 'UTC')
    # for iGHI in range(len(GHI_loc)):
    # iy0, ix0 = iys[iGHI], ixs[iGHI]
    # print("\tExtracting t=%s for %i: %i, %i from %i, %i" % (timestamp,iGHI,iy0,ix0,ny,nx))
    slc = np.s_[max(0, iy0 - WIN_SIZE):min(ny - 1, iy0 + WIN_SIZE), max(0, ix0 - WIN_SIZE):min(nx - 1, ix0 + WIN_SIZE)]

    if stch.cloud_mask[slc].size < 1:
        logging.info("\tInvalid cloud mask slice selection")
        return None
    try:
        stch.cloud_mask = stch.cloud_mask.filled(0)      #to eliminate nans in output when no clouds are present in slc
    except AttributeError:
        pass
        
    rgb0 = stch.rgb.astype(np.float32)
    rgb0[rgb0 <= 0] = np.nan
    rgb = np.reshape(rgb0[slc], (-1, 3))

    # Count the number of non-NaN elements
    nsum = np.sum(~np.isnan(rgb), axis=0)
    if nsum[0] == 0:
        logging.info("\tNaN slice encountered")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)     #expect RuntimeWarnings here for blank frames
        R_mean1, G_mean1, B_mean1 = np.nanmean(rgb, axis=0);

        R_min1, G_min1, B_min1 = np.nanmin(rgb, axis=0);
        R_max1, G_max1, B_max1 = np.nanmax(rgb, axis=0);
        RBR1 = (R_mean1 - B_mean1) / (R_mean1 + B_mean1)
        cf1 = np.sum(stch.cloud_mask[slc]) / np.sum(rgb[:, 0] > 0);

        dt_timestamp = datetime.fromtimestamp(timestamp, tz=timezone("UTC"))
        times = pd.DatetimeIndex([dt_timestamp + timedelta(minutes=lm) for lm in lead_minutes])
        # print( times )
        # unused: ghis = loc.get_clearsky( times )
        # Note: calculated values below are for the forecast time, not the current feature time
        max_ghis = list(loc.get_clearsky(times)['ghi'])
        max_dnis = list(loc.get_clearsky(times)['dni'])
        max_dhis = list(loc.get_clearsky(times)['dhi'])

        out_args = []
        for ilt, lead_time in enumerate(lead_steps):
            iy = int(0.5 + iy0 + stch.velocity[0] * lead_time)
            ix = int(0.5 + ix0 - stch.velocity[1] * lead_time)  #####  ? need to revert vx since the image is flipped
            slc = np.s_[max(0, iy - WIN_SIZE):min(ny - 1, iy + WIN_SIZE),
                  max(0, ix - WIN_SIZE):min(nx - 1, ix + WIN_SIZE)]
            if stch.cloud_mask[slc].size >= 1:                  #Future: Could RGB info still be useful without cloud mask?
                rgb = np.reshape(rgb0[slc], (-1, 3));

                nsum = np.sum(~np.isnan(rgb), axis=0)
                if (nsum[0] == 0) or (iy < 0 or ix < 0):
                    continue

                R_mean2, G_mean2, B_mean2 = np.nanmean(rgb, axis=0);

                R_min2, G_min2, B_min2 = np.nanmin(rgb, axis=0)
                R_max2, G_max2, B_max2 = np.nanmax(rgb, axis=0)
                RBR2 = (R_mean2 - B_mean2) / (R_mean2 + B_mean2)
                cf2 = np.sum(stch.cloud_mask[slc]) / np.sum(rgb[:, 0] > 0)

                tmp = np.asarray([iGHI, lead_minutes[ilt], timestamp, stch.cloud_base_height, stch.saa,
                                  cf1, R_mean1, G_mean1, B_mean1, R_min1, G_min1, B_min1, R_max1, G_max1, B_max1, RBR1,
                                  cf2, R_mean2, G_mean2, B_mean2, R_min2, G_min2, B_min2, R_max2, G_max2, B_max2, RBR2,
                                  cf_total, stch.velocity[0], stch.velocity[1], max_ghis[ilt], max_dnis[ilt], max_dhis[ilt]], dtype=np.float64)
                tmp = np.reshape(tmp, (1, -1))

                logging.debug("\t\tTimestamp: %li \tiGHI: %i \tlead_time: %i \tlead_minutes: %i, win: %s" % (timestamp, iGHI, lead_time, lead_minutes[ilt], str([max(0,iy-WIN_SIZE), min(ny-1,iy+WIN_SIZE), max(0,ix-WIN_SIZE), min(nx-1,ix+WIN_SIZE)])))
                plt_data = (ix, iy)
                plt0_data = (ix0, iy0)
                out_args += [(plt0_data, plt_data, iGHI, tmp)]

    return out_args


#############################################################################

@ray.remote
def extract_from_stitch(day, f, cfg):
    #logging.getLogger().setLevel(logging.INFO)
    # Read the stitch data object
    stch = stitch_pipeline.restore_ncdf(f)

    if stch is None:
        logging.info("No stitched image, skipping.")
        return (None, None)
    if stch.cloud_mask is None:
        logging.info("No cloud mask, skipping.")
        return (None, None)

    timestamp = utils.localToUTC(datetime.strptime(f[-17:-3], '%Y%m%d%H%M%S'), cfg.cam_tz)
    timestamp = timestamp.timestamp()

    ny, nx = stch.cloud_mask.shape
    y, x = (stch.lat - cfg.GHI_loc[:, 0]) * deg2km, (cfg.GHI_loc[:, 1] - stch.lon) * deg2km * np.cos(
        np.deg2rad(cfg.GHI_loc[0, 0]))

    iys = (0.5 + (y + stch.cloud_base_height * np.tan(stch.saa) * np.cos(stch.saz)) / stch.pixel_size).astype(
        np.int32)
    ixs = (0.5 + (x - stch.cloud_base_height * np.tan(stch.saa) * np.sin(stch.saz)) / stch.pixel_size).astype(
        np.int32)
    
    #Calculate total cloud fraction for time slice
    rgb0 = stch.rgb.astype(np.float32)
    rgb0[rgb0 <= 0] = np.nan
    cf_total = np.sum(stch.cloud_mask[:]) / np.sum(~np.isnan(np.reshape(stch.rgb,(-1,3))[:, 0]))

    args = [[iGHI, iys[iGHI], ixs[iGHI], ny, nx, stch, timestamp, cfg.outpath + day[:8], cfg.GHI_loc[iGHI], cfg.lead_minutes, cfg.lead_steps, cf_total] for iGHI in
            range(len(cfg.GHI_loc))]

    # Extract features (list of lists)
    #ft = ray.get([extract_MP.remote(arg) for arg in args])
    ft = [extract_MP(arg) for arg in args]
    # Make a flat list out of list of lists [filtering out None elements]
    features = [sublist for sublist in ft if sublist is not None]

    if len(features) == 0:
        logging.info("\tNo features found")
        return (None, None)

    #############################################################################

    if SAVE_FIG:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True);
            ax[0].imshow(stch.rgb);
            ax[1].imshow(stch.cloud_mask);
            colors = matplotlib.cm.rainbow(np.linspace(1, 0, len(cfg.lead_minutes)))
            plt.quiver([0, 0, 0], [0, 0, 0], [1, -2, 4], [1, 2, -7], angles='xy', scale_units='xy', scale=1)

    forecast_stats = np.zeros((len(cfg.GHI_loc), len(cfg.lead_minutes)))
    out_data = {}
    
    for iGHI in features:
        for idx, args in enumerate(iGHI):
            idx_GHI = args[2]
            try:
                out_data[idx_GHI] += [args[3]]
            except KeyError:
                out_data[idx_GHI] = [args[3]]
            #np.savetxt(fhs[idx_GHI], *args[3:])
            forecast_stats[idx_GHI, idx] += 1

            if SAVE_FIG:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)     #expect UserWarnings here
                    # On first index of a new point, also plot the "base" location and setup emtpy stats
                    if idx == 0:
                        ix, iy = args[0]
                        ax[0].scatter(ix, iy, s=6, marker='x', c='black', edgecolors='face');
                        ax[0].text(ix + 25, iy, str(idx_GHI), color='darkgray', fontsize='x-small')

                    ix, iy = args[1]
                    cc = colors[idx].reshape(1,
                                             -1)  # Make the color a 2D array to avoid value-mapping in case ix, iy length matches the color length (in scatter)
                    ax[0].scatter(ix, iy, s=6, marker='o', c=cc, edgecolors='face')
                    ax[0].text(ix + 25, iy, str(idx_GHI), color=colors[idx], fontsize='x-small',
                               bbox=dict(facecolor='darkgray', edgecolor=colors[idx], boxstyle='round,pad=0'))

    if SAVE_FIG:
        plt.tight_layout();
        plt.savefig(cfg.outpath + day[:8] + '/' + f[-18:-4] + '_features.png')
        # plt.show()
        plt.close()
    return (out_data, forecast_stats)


def extract(day, cfg):

    header_txt = b"location,lead_minutes,timestamp,stch.height,stch.saa,cf1,R_mean1,G_mean1,B_mean1,R_min1,G_min1,B_min1,R_max1,G_max1,B_max1,RBR1,cf2,R_mean2,G_mean2,B_mean2,R_min2,G_min2,B_min2,R_max2,G_max2,B_max2,RBR2,cf_total,velocity_x,velocity_y,max_ghi,max_dni,max_dhi\n"

    dir = cfg.outpath + day[:8]
    if not os.path.isdir(dir):
        try:
            logging.info("Creating directory " + dir)
            os.mkdir(dir)
        except OSError as e:
            logging.info("Error creating directory " + dir + ": " + str(e))
            return False

    fhs = []
    for iGHI in range(len(cfg.GHI_loc)):
        fhs += [open(cfg.outpath + day[:8] + '/GHI' + format(iGHI, '02') + '.csv', 'wb')]
        fhs[iGHI].write(header_txt)

    print("\n\nExtracting features for %s, GHI sensors:" % day)
    for ff in fhs:
        logging.info("\t" + ff.name)

    flist = sorted(glob.glob(cfg.stitch_path + day[:8] + '/' + day + '*.nc'))
    print("\tFound %i stitch files" % len(flist))

    forecast_stats = np.zeros((len(cfg.GHI_loc), len(cfg.lead_minutes)))
  
    for idx, flist_chunk in chunks(flist, feature_chunk_size):
        print("\t  Processing " + str(idx) + "-" + str(idx+feature_chunk_size) + " of " + str(len(flist)))

        results = ray.get([extract_from_stitch.remote(day, f, cfg) for f in flist_chunk])
        feature_results = [i[0] for i in results if i[0] is not None]
        stat_results = [i[1] for i in results if i[0] is not None]
        try:
            forecast_stats += sum(stat_results)  
        except TypeError:
            pass    #if stats are blank move on
        print("\t\tFrames analyzed: " + str(sum(x is not None for x in feature_results)))
        #print("\t\tKeys in feature_results: " + str([i.keys() for i in feature_results]))
        
        for idx_GHI in range(len(cfg.GHI_loc)):
            try:
                feature_data = np.vstack([np.vstack(i[idx_GHI]) for i in feature_results if idx_GHI in i.keys()])      #stack forecast intervals, then records in chunk for each location
                logging.info("\t\t\tFor loc %i, feature_data len = %i" % (idx_GHI, len(feature_data)))
                
                np.savetxt(fhs[idx_GHI], feature_data, fmt=', '.join(['%g'] + ['%g'] + ['%f'] + ['%g'] * (30))) #28 is cols in output string - 2
            except KeyError as e:
                logging.warning("Missing key for loc %d: %s" % (idx_GHI, str(e)))   
            except TypeError as e:
                logging.warning("Type error saving loc %d: %s" % (idx_GHI, str(e)))
            except ValueError as e:
                logging.warning("Value error saving loc %d: %s" % (idx_GHI, str(e)))

    for fh in fhs:
        fh.close()

    np.savetxt(cfg.outpath + day[:8] + '/forecast_stats.csv', forecast_stats, fmt="%i", delimiter=',')
    return forecast_stats

