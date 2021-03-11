import glob, os, sys
from configparser import ConfigParser
from ast import literal_eval
from pytz import timezone
import stitch
import camera
import multiprocessing

#############################################################################

SAVE_FIG = True
REPROCESS = False  # Reprocess already processed file?

#############################################################################

if __name__ == "__main__":

    print("/// Generate stitch ///")

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
        # Location of the cameras (lat/lon)
        location = literal_eval(cp["cameras"]["location"])
        # Parameters
        param = literal_eval(cp["cameras"]["parameters"])

        # List of cameras IDs + stitch pair (without repeated elements)
        cid_flat = cameras_id + [stitch_pair[camID] for camID in cameras_id]
        # Forecast days
        days = literal_eval(cp["forecast"]["days"])

        # Paths
        tmpfs = literal_eval(cp["paths"]["tmpfs"])
        stitch_path = literal_eval(cp["paths"]["stitch_path"])
        static_mask_path = literal_eval(cp["paths"]["static_mask_path"])

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

    # Allow interactive plots during debugging
    # plt.ioff()  #Turn off interactive plotting for running automatically

    #############################################################################

    # Initialize the list of camera objects (only once)
    cameras = {};

    if not os.path.isdir(static_mask_path):
        try:
            print("Creating directory " + static_mask_path)
            os.mkdir(static_mask_path)
        except OSError:
            print("Cannot create directory " + static_mask_path)
            exit()

    for camID in all_cameras:
        lat = location[camID][0]
        lon = location[camID][1]

        nx0 = ny0 = param[camID][0]
        xstart = param[camID][2]
        ystart = param[camID][1]
        rotation = param[camID][3]
        beta = param[camID][4]
        azm = param[camID][5]
        c1 = param[camID][6]
        c2 = param[camID][7]
        c3 = param[camID][8]

        cameras[camID] = camera.camera(camID, lat, lon, nx0, ny0, \
                                       xstart, ystart, rotation, \
                                       beta, azm, c1, c2, c3, \
                                       max_theta=70, \
                                       nxy=1000, \
                                       timezone=cam_tz, \
                                       fpath=static_mask_path)

    #############################################################################

    # Prepare the multithread pool
    print("Number of cores to use: %s" % cores_to_use)

    if cores_to_use > 1:
        pool = multiprocessing.Pool(cores_to_use, maxtasksperchild=128)

    print("DAYS: %s" % days)

    for day in days:

        dir = stitch_path + day[:8]
        if not os.path.isdir(dir):
            try:
                print("Creating directory " + dir)
                os.mkdir(dir)
            except OSError:
                print("Cannot create directory " + dir)
                continue

        print("Running Generate Stich for: %s" % day)
        ymd = day[:8]

        #############################################################################

        # Generate stitch (using pairs of cameras)

        selected = []
        img_dict = {}
        for cid in all_cameras:
            if cid not in cid_flat or stitch_pair[cid] in selected:
                continue;

            selected += cid
            ncdfs = sorted(glob.glob(tmpfs + day[:8] + '/' + cid + '_' + day + '*.nc'));
            print("\t%s: %i netcdf files found" % (cid, len(ncdfs)))
            for ncdf in ncdfs:
                try:
                    img_dict[ncdf[-17:-3]].update({cid: ncdf})
                except KeyError:
                    img_dict[ncdf[-17:-3]] = {cid: ncdf}

        print("\n\tUsing %i cores to process %i image times" % (cores_to_use, len(img_dict)))

        args = [[stitch_path, btimes, cam_list, cameras, REPROCESS, SAVE_FIG] for btimes, cam_list in img_dict.items()]

        #############################################################################
        # Debugging purposes
        #count = 0
        #for arg in args:
        #    if arg[1] == "20181201160811":
        #        break;
        #    count = count + 1
        #args = [args[count]]
        #cores_to_use = 1
        #args[0][4] = True
        #############################################################################

        # Stitch generation
        if cores_to_use > 1:
            pool.map(stitch.generate_stitch, args, chunksize=16)
        else:
            for arg in args:
                stitch.generate_stitch(arg)

    # End of generate_stitch
    if cores_to_use > 1:
        pool.close()
        pool.join()

    exit()