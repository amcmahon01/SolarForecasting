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
import stitch
import utils

#############################################################################

deg2km = 6367 * np.pi / 180
WIN_SIZE = 50  # half-width of bounding box integrated per GHI point
INTERVAL = 0.5  # 0.5 min or 30 sec

SAVE_FIG = True
REPROCESS = False  # Not yet implemented

#############################################################################

def extract_MP(args):
    # TODO lead_steps should be input of the function

    iGHI, iy0, ix0, ny, nx, stch, timestamp, outpath, latlon, lead_minutes, lead_steps = args

    if (iy0 < 0) or (ix0 < 0):
        print("\tInvalid dimensions")
        return None

    loc = Location(latlon[0], latlon[1], 'UTC')
    # for iGHI in range(len(GHI_loc)):
    # iy0, ix0 = iys[iGHI], ixs[iGHI]
    # print("\tExtracting t=%s for %i: %i, %i from %i, %i" % (timestamp,iGHI,iy0,ix0,ny,nx))
    slc = np.s_[max(0, iy0 - WIN_SIZE):min(ny - 1, iy0 + WIN_SIZE), max(0, ix0 - WIN_SIZE):min(nx - 1, ix0 + WIN_SIZE)]

    if stch.cloud_mask[slc].size < 1:
        # print("\tInvalid cloud mask slice selection")
        return None

    rgb0 = stch.rgb.astype(np.float32)
    rgb0[rgb0 <= 0] = np.nan
    rgb = np.reshape(rgb0[slc], (-1, 3))

    # Count the number of non-NaN elements
    nsum = np.sum(~np.isnan(rgb), axis=0)
    if nsum[0] == 0:
        # print("\tNaN slice encountered")
        return None

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
    cf_total = np.sum(stch.cloud_mask[:]) / np.sum(rgb[:, 0] > 0)

    out_args = []
    for ilt, lead_time in enumerate(lead_steps):
        iy = int(0.5 + iy0 + stch.velocity[0] * lead_time)
        ix = int(0.5 + ix0 - stch.velocity[1] * lead_time)  #####  ? need to revert vx since the image is flipped
        slc = np.s_[max(0, iy - WIN_SIZE):min(ny - 1, iy + WIN_SIZE),
              max(0, ix - WIN_SIZE):min(nx - 1, ix + WIN_SIZE)]
        if stch.cloud_mask[slc].size >= 1:
            rgb = np.reshape(rgb0[slc], (-1, 3));

            nsum = np.sum(~np.isnan(rgb), axis=0)
            if (nsum[0] == 0) or (iy < 0 or ix < 0):
                continue

            R_mean2, G_mean2, B_mean2 = np.nanmean(rgb, axis=0);

            R_min2, G_min2, B_min2 = np.nanmin(rgb, axis=0)
            R_max2, G_max2, B_max2 = np.nanmax(rgb, axis=0)
            RBR2 = (R_mean2 - B_mean2) / (R_mean2 + B_mean2)
            cf2 = np.sum(stch.cloud_mask[slc]) / np.sum(rgb[:, 0] > 0)

            tmp = np.asarray([lead_minutes[ilt], timestamp, stch.cloud_base_height, stch.saa,
                              cf1, R_mean1, G_mean1, B_mean1, R_min1, G_min1, B_min1, R_max1, G_max1, B_max1, RBR1,
                              cf2, R_mean2, G_mean2, B_mean2, R_min2, G_min2, B_min2, R_max2, G_max2, B_max2, RBR2,
                              cf_total, max_ghis[ilt], max_dnis[ilt], max_dhis[ilt]], dtype=np.float64)
            tmp = np.reshape(tmp, (1, -1))

            # print("\t\tTimestamp: %li \tiGHI: %i \tlead_time: %i \tlead_minutes: %i, win: %s" % (timestamp, iGHI, lead_time, lead_minutes[ilt], str([max(0,iy-WIN_SIZE), min(ny-1,iy+WIN_SIZE), max(0,ix-WIN_SIZE), min(nx-1,ix+WIN_SIZE)])))
            plt_data = (ix, iy)
            plt0_data = (ix0, iy0)
            out_args += [(plt0_data, plt_data, iGHI, tmp, ', '.join(['%g'] + ['%f'] + ['%g'] * (tmp.size - 2)))]

    return out_args


#############################################################################

if __name__ == "__main__":

    print("/// Extract features ///")

    #############################################################################
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
        all_cameras = literal_eval(cp["cameras"]["all_cameras"])
        # Pairs of cameras (stitch criteria)
        stitch_pair = literal_eval(cp["cameras"]["stitch_pair"])
        # IDs
        cameras_id = literal_eval(cp["cameras"]["cameras_id"])

        # List of cameras IDs + stitch pair (without repeated elements)
        cid_flat = cameras_id + [stitch_pair[camID] for camID in cameras_id]
        # Forecast days
        days = literal_eval(cp["forecast"]["days"])

        # Paths
        outpath = literal_eval(cp["paths"]["feature_path"])
        stitch_path = literal_eval(cp["paths"]["stitch_path"])

        GHI_Coor = literal_eval(cp["GHI_sensors"]["GHI_Coor"])
        GHI_loc = [GHI_Coor[key] for key in sorted(GHI_Coor)]
        GHI_loc = np.array(GHI_loc)

        lead_minutes = literal_eval(cp["forecast"]["lead_minutes"])
        lead_steps = [lt / INTERVAL for lt in lead_minutes]
        days = literal_eval(cp["forecast"]["days"])

        # Define time zone (EST)
        try:
            cam_tz = timezone(cp["cameras"]["timezone"])
            print("Using camera timezone: %s" % str(cam_tz))
        except Exception:
            cam_tz = timezone("utc")
            print("Error processsing camera timezone config, assuming UTC")

        # Define number of cores
        try:
            cores_to_use = int(cp["server"]["cores_to_use"])
        except Exception:
            cores_to_use = 20

    except KeyError as e:
        print("Error loading config: %s" % e)
        exit()

    #############################################################################

    # Prepare the multithread pool
    print("Number of cores to use: %s" % cores_to_use)

    if cores_to_use > 1:
        pool = get_context("spawn").Pool(cores_to_use, maxtasksperchild=128)

    header_txt = b"lead_minutes,timestamp,stch.height,stch.saa,cf1,R_mean1,G_mean1,B_mean1,R_min1,G_min1,B_min1,R_max1,G_max1,B_max1,RBR1,cf2,R_mean2,G_mean2,B_mean2,R_min2,G_min2,B_min2,R_max2,G_max2,B_max2,RBR2,cf_total,max_ghi,max_dni,max_dhi\n"
    forecast_stats_total = np.zeros((len(GHI_loc), len(lead_minutes)))

    print("DAYS: %s" % days)

    for day in days:

        dir = outpath + day[:8]
        if not os.path.isdir(dir):
            try:
                print("Creating directory " + dir)
                os.mkdir(dir)
            except OSError as e:
                print("Error creating directory " + dir + ": " + str(e))
                continue

        fhs = []
        for iGHI in range(len(GHI_loc)):
            fhs += [open(outpath + day[:8] + '/GHI' + format(iGHI, '02') + '.csv', 'wb')]
            fhs[iGHI].write(header_txt)

        print("Extracting features for %s, GHI sensors:" % day)
        for ff in fhs:
            print("\t" + ff.name)

        flist = sorted(glob.glob(stitch_path + day[:8] + '/' + day + '*.nc'))
        print("\tFound %i stitch files" % len(flist))

        forecast_stats = np.zeros((len(GHI_loc), len(lead_minutes)))

        for f in flist:

            # Read the stitch data object
            stch = stitch.restore_ncdf(f)

            if stch is None:
                print("No stitched image, skipping.")
                continue
            if stch.cloud_mask is None:
                print("No cloud mask, skipping.")
                continue

            timestamp = utils.localToUTC(datetime.strptime(f[-17:-3], '%Y%m%d%H%M%S'), cam_tz)
            timestamp = timestamp.timestamp()

            ny, nx = stch.cloud_mask.shape
            y, x = (stch.lat - GHI_loc[:, 0]) * deg2km, (GHI_loc[:, 1] - stch.lon) * deg2km * np.cos(
                np.deg2rad(GHI_loc[0, 0]))

            iys = (0.5 + (y + stch.cloud_base_height * np.tan(stch.saa) * np.cos(stch.saz)) / stch.pixel_size).astype(
                np.int32)
            ixs = (0.5 + (x - stch.cloud_base_height * np.tan(stch.saa) * np.sin(stch.saz)) / stch.pixel_size).astype(
                np.int32)

            args = [[iGHI, iys[iGHI], ixs[iGHI], ny, nx, stch, timestamp, outpath + day[:8], GHI_loc[iGHI], lead_minutes, lead_steps] for iGHI in
                    range(len(GHI_loc))]

            # Extract features (list of lists)
            if cores_to_use > 1:
                ft = pool.map(extract_MP, args, chunksize=16)
                # Make a flat list out of list of lists [filtering out None elements]
                features = [sublist for sublist in ft if sublist is not None]
            else:
                features = []
                for arg in args:
                    ft = extract_MP(arg)
                    if ft is not None:
                        features += [ft]

            if len(features) == 0:
                print("\tNo features found")
                continue

            #############################################################################

            if SAVE_FIG:
                fig, ax = plt.subplots(1, 2, sharex=True, sharey=True);
                ax[0].imshow(stch.rgb);
                ax[1].imshow(stch.cloud_mask);
                colors = matplotlib.cm.rainbow(np.linspace(1, 0, len(lead_minutes)))
                plt.quiver([0, 0, 0], [0, 0, 0], [1, -2, 4], [1, 2, -7], angles='xy', scale_units='xy', scale=1)

            for iGHI in features:
                for idx, args in enumerate(iGHI):
                    idx_GHI = args[2]

                    np.savetxt(fhs[idx_GHI], *args[3:])
                    forecast_stats[idx_GHI, idx] += 1

                    if SAVE_FIG:
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
                plt.savefig(outpath + day[:8] + '/' + f[-18:-4] + '_features.png')
                # plt.show()
                plt.close()

        for fh in fhs:
            fh.close()

        forecast_stats_total += forecast_stats
        np.savetxt(outpath + day[:8] + '/forecast_stats.csv', forecast_stats, fmt="%i", delimiter=',')

    if cores_to_use > 1:
        pool.close()
        pool.join()
    
    np.savetxt(outpath + 'forecast_stats_run_' + day[:8] + '.csv', forecast_stats_total, fmt="%i", delimiter=',')