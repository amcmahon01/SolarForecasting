import glob, os, sys
from configparser import ConfigParser
from ast import literal_eval
from pytz import timezone
import camera
import image
import multiprocessing

#############################################################################

REPROCESS = False  # Reprocess already processed file?

#############################################################################

if __name__ == "__main__":

    print("/// Pre-processing ///")

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
        # The cameras are grouped in groupes to determine height
        height_group = literal_eval(cp["cameras"]["height_group"])
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
        inpath = literal_eval(cp["paths"]["inpath"])
        tmpfs = literal_eval(cp["paths"]["tmpfs"])
        static_mask_path = literal_eval(cp["paths"]["static_mask_path"])

        # Define time zone (EST)
        try:
            cam_tz = timezone(cp["cameras"]["timezone"])
            print("Using camera timezone: %s" % str(cam_tz))
        except Exception:
            cam_tz = timezone("utc")
            print("Error processing camera timezone config, assuming UTC")

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

    # TODO improve map delay time between chunksize calls

    print("DAYS: %s" % days)

    for day in days:

        dir = tmpfs + day[:8]
        if not os.path.isdir(dir):
            try:
                print("Creating directory "+dir)
                os.mkdir(dir)
            except OSError:
                print("Cannot create directory "+dir)
                continue

        print("Running Preprocessing for: %s" % day)
        ymd = day[:8]

        #############################################################################

        # Create all the images
        for cid in cid_flat:

            camera_cur = cameras[cid]

            # List of files
            print('\t' + inpath + camera_cur.camID + '/' + ymd + '/')
            flist = sorted(glob.glob(inpath + camera_cur.camID + '/' + ymd + '/' + camera_cur.camID + '_' + day + '*jpg'))

            print("\tFound %i image files for %s" % (len(flist), camera_cur.camID))
            if len(flist) <= 0:
                continue

            args = [[camera_cur, f, tmpfs, REPROCESS] for f in flist]

            # Create all images of the current camera, process the cloud mask and cloud motion
            if cores_to_use > 1:
                pool.map(image.process_image,        args, chunksize=16)
                pool.map(image.process_bright_mask,  args, chunksize=16)
                pool.map(image.process_cloud_mask,   args, chunksize=16)
                pool.map(image.process_cloud_motion, args, chunksize=16)
            else:
                for arg in args:
                    image.process_image(arg)
                for arg in args:
                    image.process_bright_mask(arg)
                for arg in args:
                    image.process_cloud_mask(arg)
                for arg in args:
                    image.process_cloud_motion(arg)

        # Compute cloud base height (using pairs of cameras)
        for cid in cid_flat:
            camera_cur = cameras[cid]

            # List of files
            print('\t' + inpath + camera_cur.camID + '/' + ymd + '/')
            flist = sorted(glob.glob(inpath + camera_cur.camID + '/' + ymd + '/' + camera_cur.camID + '_' + day + '*jpg'))

            print("\tFound %i image files for %s" % (len(flist), camera_cur.camID))
            if len(flist) <= 0:
                continue

            for cmr in height_group[cid]:
                camera_ngh = cameras[cmr]

                args = [[camera_cur, camera_ngh, f, tmpfs, REPROCESS] for f in flist]
                if cores_to_use > 1:
                    pool.map(image.process_cloud_height, args, chunksize=16)
                else:
                    for arg in args:
                        image.process_cloud_height(arg)

    # End of preprocessing
    if cores_to_use > 1:
        pool.close()
        pool.join()