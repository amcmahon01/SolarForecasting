import glob, os, sys
from configparser import ConfigParser
from ast import literal_eval
from pytz import timezone
import multiprocessing
from scipy.ndimage import morphology
from skimage.morphology import remove_small_objects
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import camera
import image


def mask_from_image(f, cam):

    img=image.image(f, cam);  ###img object contains four data fields: rgb, red, rbr, and cm 
    img.read_image()
    img.undistort_image(skip_sun=True);    ###undistortion
    if img.rgb is None:
        return

    mask=(img.rgbu[:,:,2]<127)
    mask=morphology.binary_erosion(mask)     #Remove undistortion artifacts
    mask=remove_small_objects(mask, min_size=10, connectivity=4)
    mask=morphology.binary_dilation(mask,np.ones((7,7))) 
    mask=morphology.binary_closing(mask,np.ones((5,5))) 
    mask=remove_small_objects(mask, min_size=1000, connectivity=4)

    fig,ax=plt.subplots(2,2,sharex=True,sharey=True);     
    ax[0,0].imshow(img.rgbu)
    ax[0,0].set_title('RGB Undistorted')

    ax[0,1].imshow(img.red,vmin=-0.2,vmax=0.1)
    ax[0,1].set_title('Red Thresholds')

    ax[1,0].imshow(mask)
    ax[1,0].set_title('Static Mask')

    ax[1,1].imshow(img.rgbu[:,:,2]) 
    ax[1,1].set_title('Red Channel')
#     plt.figure(); plt.hist(img.rbr[img.rbr>-1],bins=100);
    
    plt.savefig(os.path.join(os.path.split(f)[0], cam.camID + "_masks.png"))
    #plt.show()

    np.save(os.path.join(os.path.split(f)[0], cam.camID + '_mask'), mask)
    return




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
        cid_flat = cameras_id + [stitch_pair[camID] for camID in cameras_id if stitch_pair[camID] is not '']

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
        print("Error loading config: %s" % str(e))
        exit()


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

        print("Loading " + camID)

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
                                       fpath=static_mask_path, \
                                       ignore_mask=True)

    #############################################################################

    # Check for masks and jpeg masks
    for cid in cid_flat:
        print("Checking mask for " + cid)
        camera_cur = cameras[cid]
        fname_mask = static_mask_path + camera_cur.camID + '_mask.npy'
        fname_mask_jpeg = static_mask_path + camera_cur.camID  + '_mask.jpg'

        # Check for existing mask
        if not os.path.isfile(fname_mask):
            print("\tNo existing mask found, checking for jpeg mask")
            mask_from_image(fname_mask_jpeg, camera_cur)

    print("Done updating masks, recomputing cameras...")

    cameras = {};
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
                                       fpath=static_mask_path, \
                                       overwrite=True)              #Create new object, overwrites old one!

    print("Done updating cameras and masks.")