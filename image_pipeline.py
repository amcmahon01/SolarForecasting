from typing import Tuple
import numpy as np
import warnings
import stat_tools as st
from datetime import datetime, timedelta
import os, ephem
# https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_objects
from skimage.morphology import remove_small_objects
from scipy.ndimage import morphology
from scipy.ndimage.filters import maximum_filter
import matplotlib
matplotlib.use('agg')
import mncc
import geo
import pickle
from matplotlib import pyplot
import utils
import netCDF4 as nc4
import logging
import ray
import warnings
#import gc

def ignore_warnings(function):
    def wrapper(*args,**kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return function(*args,**kwargs)


# Image and camera classes for prev and neigh with reduced memory footprint
class CamShare():
    def __init__(self, id=None, lat=None, lon=None, max_theta=None, nx=None, ny=None, cos_th=None, cos_p=None, sin_p=None):
        self.camID = id
        self.lat = lat
        self.lon = lon
        self.max_theta = max_theta
        self.nx = nx
        self.ny = ny
        self.cos_th = cos_th
        self.cos_p = cos_p
        self.sin_p = sin_p

class ImageShare():      
    def __init__(self, filename=None, rgbu=None, red=None, cloud_mask=None, bright_mask=None, velocity=[], saa=None, saz=None, day_time=True, cam=None):
        self.filename = filename
        self.rgbu = rgbu
        self.red = red
        self.cloud_mask = cloud_mask
        self.bright_mask = bright_mask
        self.cloud_base_height = []
        self.height_neighbours = []
        self.velocity = velocity
        self.saa = saa
        self.saz = saz
        self.day_time = day_time
        self.camera = cam
        
# The image class
@ray.remote
class Image(object):

    def __init__(self, filename, camera):

        # self.t_local = None # not used
        self.filename = filename
        self.camera = camera
        # -1=undefined / 0=night / 1=day
        self.day_time = -1
        # Original RGB channels (after ROI)
        self.rgb = None
        # RGB channels after undistortion
        self.rgbu = None
        # Mask to filter out small bright objects (logical converted to uint8)
        self.bright_mask = None

        # Spatial structure/texture of the red image, used by the cloud motion and height routines
        # Defined in undistort
        self.red = None

        # Not sure about his attribute yet
        self.layers = 0

        # Parameters defined in undistort()
        self.saa = None
        self.saz = None
        # Cloud mask (logical converted to uint8)
        self.cloud_mask = None
        # Sun
        # self.sun_x = None
        # self.sun_y = None
        # Cloud base height (list)
        self.cloud_base_height = []
        self.height_neighbours = []
        # Velocity (list)
        self.velocity = []
        
    """
        Create minimalist class of image for prev/neighbor calcs
    """
    @ray.method(num_returns=1)
    def get_image_min(self, include_rgbu=False):
    
        if include_rgbu:
            cam_min = CamShare(self.camera.camID, self.camera.lat, self.camera.lon, self.camera.max_theta, self.camera.nx, self.camera.ny, self.camera.cos_th, self.camera.cos_p, self.camera.sin_p)
            img_min = ImageShare(self.filename, self.rgbu, self.red, self.cloud_mask, self.bright_mask, self.velocity, self.saa, self.saz, self.day_time, cam_min)
        else:
            cam_min = CamShare(self.camera.camID, self.camera.lat, self.camera.lon, self.camera.max_theta, self.camera.nx, self.camera.ny, self.camera.cos_th)
            img_min = ImageShare(self.filename, None, self.red, self.cloud_mask, None, self.velocity, self.saa, self.saz, self.day_time, cam_min)

        return img_min
        
    """
        Read minimalist class of image for prev/neighbor calcs
    """
    @ray.method(num_returns=1)
    def set_image_min(self, img_min):
    
        try:
            self.filename = img_min.filename
            self.rgbu = img_min.rgbu
            self.red = img_min.red
            self.cloud_mask = img_min.cloud_mask
            self.bright_mask = img_min.bright_mask
            self.cloud_base_height = img_min.cloud_base_height
            self.height_neighbours = img_min.height_neighbours
            self.velocity = img_min.velocity
            self.saa = img_min.saa
            self.saz = img_min.saz
            self.day_time = img_min.day_time
            self.camera = img_min.camera
        except Exception as e:
            logging.warning("Error reading min image object: " + str(e))
            raise e
            return False
        return True


    """
        Reads the image from the input file and selects the region of interest defined in the camera object
    """
    @ray.method(num_returns=1)
    def read_image(self):

        try:
            logging.info("\tReading image file " + self.filename)
            im0 = pyplot.imread(self.filename)
        except:
            logging.error('\tCannot read file ' + self.filename)
            return None

        # Select region of interest
        try:
            self.rgb = im0[self.camera.roi]
        except:
            logging.warning('\tCannot select region of interest')
            return None
        
        return self.filename

    """
        Saves the image object in a Pickle file (deprecated)
    """

    # https://docs.python.org/2.0/lib/module-pickle.html
    def save_pickle(self, filename):

        logging.info("\tSaving image file " + filename)
        try:
            with open(filename, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        except:
            logging.error("\tCannot save the file " + filename)

    """
        Save the attributes of the image object in a NetCDF file
    """
    @ray.method(num_returns=1)
    def save_netcdf(self, filename, inmemory=False):

        try:
            if inmemory:
                logging.info("\tSaving in mem: " + str(filename))
                memory = 1024
            else:
                logging.info("\tSaving image file " + str(filename))
                memory = None

            if self.day_time == 0:
                logging.info("Night time, skipping " + str(filename))
                return False

            root_grp = nc4.Dataset(filename, mode='w',memory=memory)  # 'w' stands for write, memory size doesnt't matter using netcdf v4
            root_grp.description = 'SolarForecasting image object representation'
            root_grp.file_source = str(self.filename)  # os.path.basename(self.filename)
            root_grp.history = "Created on " + datetime.utcnow().strftime("%b-%d-%Y %H:%M:%S")

            # Dimensions
            root_grp.createDimension('one', 1)

            if (self.rgbu is not None):
                nx, ny, nz = self.rgbu.shape
                root_grp.createDimension('x', nx)
                root_grp.createDimension('y', ny)
                root_grp.createDimension('z', nz)

            # Variables (with zlib compression, level=9)
            daytime = root_grp.createVariable('DayTime', 'i4', ('one',))
            daytime.description = "Time of the day (-1=undefined, 0=night, 1=day)"
            daytime[:] = self.day_time

            # What about self.sun_x, self.sun_y?
            if self.saa is not None:
                saa = root_grp.createVariable('SolarAltitudeAngle', 'f4', 'one')
                saa.description = "Solar altitude angle"
                saa[:] = self.saa
            if self.saz is not None:
                saz = root_grp.createVariable('SolarAzimuthAngle', 'f4', 'one')
                saz.description = "Solar azimuth angle"
                saz[:] = self.saz
            if self.rgbu is not None:
                rgbu = root_grp.createVariable('RGBu', self.rgbu.dtype.str, ('x', 'y', 'z'), zlib=True, complevel=9)
                rgbu.description = "Undistorted RGB channels"
                rgbu[:, :, :] = self.rgbu
            if self.bright_mask is not None:
                bmsk = root_grp.createVariable('BrightMask', self.bright_mask.dtype.str, ('x', 'y'), zlib=True, complevel=9, fill_value=0)
                bmsk.description = "Mask of small bright objects"
                bmsk[:, :] = self.bright_mask
            if self.red is not None:
                redu = root_grp.createVariable('Red', self.red.dtype.str, ('x', 'y'), zlib=True, complevel=9)
                redu.description = "Spatial texture of the red channel, used to determine cloud motion and cloud base height"
                redu[:, :] = self.red
            if self.cloud_mask is not None:
                cmsk = root_grp.createVariable('CloudMask', self.cloud_mask.dtype.str, ('x', 'y'), zlib=True, complevel=9, fill_value=0)
                cmsk.description = "Cloud mask"
                cmsk[:, :] = self.cloud_mask
            if len(self.velocity) > 0 and None not in self.velocity[0]:
                vel = np.asarray(self.velocity)
                vx, vy = vel.shape
                root_grp.createDimension('vx', vx)
                root_grp.createDimension('vy', vy)
                #print("CM save: " + filename + " : " + str(vel) + " : " + str(vel.dtype))
                cm = root_grp.createVariable('CloudMotion', vel.dtype.str, ('vx', 'vy'), zlib=True, complevel=9)
                cm.description = "Cloud motion velocity"
                cm[:, :] = vel
            if len(self.cloud_base_height) > 0:
                hgt = np.asarray(self.cloud_base_height)
                root_grp.createDimension('unlimited', None)
                height = root_grp.createVariable('CloudBaseHeight', hgt.dtype.str, 'unlimited', zlib=True, complevel=9)
                height.description = "First detected cloud base height [m]"
                height.neighbours = ",".join(self.height_neighbours)
                height.units = "m"
                height[:] = hgt

            if inmemory:
                data = root_grp.close()
                return data.tobytes()
            else:
                root_grp.close()
                return True
        except Exception as e:
            logging.error("\tAn error occurred creating the NetCDF file " + str(filename) + ": " + str(e))
            try:
                root_grp.close()
            except Exception:
                pass
            return False
            
    """
        Restores an image object from a NetCDF file
    """
    @ray.method(num_returns=1)
    def restore_ncdf(self, filename=None, memory=None, ignore_missing=False):

        if filename is None:
            filename = self.filename
        logging.info("\tReading image file " + str(self.filename))
        existing_data = []
        
        try:
            root_grp = nc4.Dataset(filename, 'r', memory=memory)  # 'r' stands for read
            existing_data += ["img"]
            
            if 'file_source' in root_grp.ncattrs():
                file_source = root_grp.getncattr('file_source')
            else:
                file_source = ''

            if 'DayTime' in root_grp.variables:
                self.day_time = root_grp.variables['DayTime'][0]
            if 'SolarAltitudeAngle' in root_grp.variables:
                self.saa = root_grp.variables['SolarAltitudeAngle'][0]
            if 'SolarAzimuthAngle' in root_grp.variables:
                self.saz = root_grp.variables['SolarAzimuthAngle'][0]
            if 'RGBu' in root_grp.variables:
                self.rgbu = root_grp.variables['RGBu'][:]
                if self.rgbu is not None:
                    existing_data += ["rgbu"]
            if 'BrightMask' in root_grp.variables:
                self.bright_mask = root_grp.variables['BrightMask'][:]
                if self.bright_mask is not None:
                    existing_data += ["bright_mask"]
            if 'Red' in root_grp.variables:
                self.red = root_grp.variables['Red'][:]
                if self.red is not None:
                    existing_data += ["red"]
            if 'CloudMask' in root_grp.variables:
                self.cloud_mask = root_grp.variables['CloudMask'][:]
                if self.cloud_mask is not None:
                    existing_data += ["cloud_mask"]
            if 'CloudMotion' in root_grp.variables:
                self.velocity = root_grp.variables['CloudMotion'][:].tolist()
                #print("CM restore: " + filename + " : " + str(self.velocity))
                try:
                    if None not in self.velocity[0]:
                        existing_data += ["cloud_motion"]
                except Exception:
                    pass
            if 'CloudBaseHeight' in root_grp.variables:
                self.cloud_base_height = root_grp.variables['CloudBaseHeight'][:].tolist()
                self.height_neighbours = list(root_grp.variables['CloudBaseHeight'].getncattr('neighbours').split(','))
                if self.cloud_base_height is not None:
                    existing_data += ["cloud_base_height"]

            root_grp.close()

        except Exception as e:
            #raise e
            if ignore_missing:
                logging.info("\tAn error occurred reading the NetCDF file " + str(filename) + ": " + str(e))
            else:
                logging.error("\tAn error occurred reading the NetCDF file " + str(filename) + ": " + str(e))
            #return False
            try:
                root_grp.close()
            except Exception:
                pass

        return existing_data
        

    """
        Undistort the raw image, set RGBu and red
    """
    @ray.method(num_returns=1)
    def undistort_image(self, day_only=True, skip_sun=False):

        if self.rgb is None:
            logging.warning("\tCannot undistort the image if the RGB channel is not defined")
            return False

        red0 = self.rgb[:, :, 0].astype(np.float32);
        red0[red0 <= 0] = np.nan

        if skip_sun == False:
            # Get the image acquisition time, this need to be adjusted whenever the naming convention changes
            t_std = utils.localToUTC(datetime.strptime(self.filename[-18:-4], '%Y%m%d%H%M%S'), self.camera.timezone)

            logging.info("\tUndistortion %s" % (str(t_std)))
            gatech = ephem.Observer();
            gatech.date = t_std.strftime('%Y/%m/%d %H:%M:%S')
            gatech.lat = str(self.camera.lat)
            gatech.lon = str(self.camera.lon)
            sun = ephem.Sun();
            sun.compute(gatech);

            # Sun parameters
            self.saa = np.pi / 2 - sun.alt;
            self.saz = np.deg2rad((180 + np.rad2deg(sun.az)) % 360);

            # if False:
            if day_only and self.saa > np.deg2rad(75):
                logging.info("\tNight time (sun angle = %f), skipping" % self.saa)
                self.day_time = 0
                return False
            else:
                logging.info("\tDay time (sun angle = %f)" % self.saa)
                self.day_time = 1

            cos_sz = np.cos(self.saa)
            cos_g = cos_sz * np.cos(self.camera.theta0) + np.sin(self.saa) * np.sin(self.camera.theta0) * np.cos(
                self.camera.phi0 - self.saz);

            # RuntimeWarnings expected in this block
            if np.nanmean(red0[(cos_g > 0.995) & (red0 >= 1)]) > 230:
                mk = cos_g > 0.98
                red0[mk] = np.nan

        # Not used ??
        # xsun = np.tan(self.saa) * np.sin(saz)
        # ysun = np.tan(self.saa) * np.cos(saz)

        # self.sun_x = int(0.5 * self.camera.nx * (1 + xsun / self.camera.max_tan))
        # self.sun_y = int(0.5 * self.camera.ny * (1 + ysun / self.camera.max_tan))

        invalid = ~self.camera.valid

        red = st.fast_bin_average2(red0, self.camera.weights);
        red = st.fill_by_mean2(red, 7, mask=(np.isnan(red)) & self.camera.valid)
        red[invalid] = np.nan;

        #  plt.figure(); plt.imshow(red); plt.show();

        red -= st.rolling_mean2(red, int(self.camera.nx // 6.666))

        # RuntimeWarnings expected in this block
        red[red > 50] = 50;
        red[red < -50] = -50
        red = (red + 50) * 2.54 + 1;

        red[invalid] = 0

        self.red = red.astype(np.uint8)

        im = np.zeros((self.camera.ny, self.camera.nx, 3), dtype=self.rgb.dtype)
        for i in range(3):
            im[:, :, i] = st.fast_bin_average2(self.rgb[:, :, i], self.camera.weights);
            im[:, :, i] = st.fill_by_mean2(im[:, :, i], 7, ignore=0, mask=(im[:, :, i] == 0) & (self.camera.valid))
        im[self.red <= 0] = 0

        self.rgbu = im
       
        return True

    """
      Computes the bright mask to filter out small bright objects      
    """
    @ray.method(num_returns=1)
    def compute_bright_mask(self, img_prev):

        if (self.red is None) or (img_prev.red is None):
            if (img_prev.red is None): #if only prev is missing, this is likely the first image in a time series and this is expected
                logging.info("\tCannot remove small objects on \"distorted\" images")
            else:
                logging.warning("\tCannot remove small objects on \"distorted\" images")
            return False

        r0 = img_prev.red.astype(np.float32)
        r0[r0 <= 0] = np.nan
        r1 = self.red.astype(np.float32)
        r1[r1 <= 0] = np.nan

        err0 = r1 - r0

        dif = np.abs(err0);
        dif = st.rolling_mean2(dif, 20)
        semi_static = (abs(dif) < 10) & (r0 - 127 > 100)
        semi_static = morphology.binary_closing(semi_static, np.ones((10, 10)))
        semi_static = remove_small_objects(semi_static, min_size=200, in_place=True)

        self.bright_mask = semi_static.astype(np.uint8)
        return True

    """
      Computes the cloud mask
    """
    @ray.method(num_returns=1)
    def compute_cloud_mask(self, img_prev):

        try:
            if (self.rgbu is None) or (img_prev.rgbu is None):
                if (img_prev.rgbu is None):
                    logging.info("\tCannot compute cloud mask on \"distorted\" images")
                else:
                    logging.warning("\tCannot compute cloud mask on \"distorted\" images")
                return False

            if self.bright_mask is None:
                logging.warning("\tCannot compute cloud mask on images where the bright mask has not been defined")
                return False

            self.cloud_mask = np.full((1000,1000),dtype=np.uint8,fill_value=0)

            # RGB images
            rgb_curr = self.rgbu
            rgb_prev = img_prev.rgbu

            # Remove small bright objects (only in the current image)
            semi_static = self.bright_mask > 0
            rgb_curr[semi_static] = 0

            cos_s = np.cos(self.saa);
            sin_s = np.sin(self.saa)
            cos_sp = np.cos(self.saz);
            sin_sp = np.sin(self.saz)
            cos_th = self.camera.cos_th;
            sin_th = np.sqrt(1 - cos_th ** 2)
            cos_p = self.camera.cos_p;
            sin_p = self.camera.sin_p
            # Cosine of the angle between illumination and view directions
            cos_g = cos_s * cos_th + sin_s * sin_th * (cos_sp * cos_p + sin_sp * sin_p);

            # Previous image
            r0 = rgb_prev[..., 0].astype(np.float32);
            r0[r0 <= 0] = np.nan
            # Current image (self)
            r1 = rgb_curr[..., 0].astype(np.float32);
            r1[r1 <= 0] = np.nan

            rbr_raw = (r1 - rgb_curr[:, :, 2]) / (rgb_curr[:, :, 2] + r1)
            rbr = rbr_raw.copy();
            rbr -= st.rolling_mean2(rbr, int(self.camera.nx // 6.666))

            rbr[rbr >  0.08] =  0.08;
            rbr[rbr < -0.08] = -0.08;

            # Scale rbr to 0-255                      
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                rbr = (rbr + 0.08) * 1587.5 + 1;
                mblue = np.nanmean(rgb_curr[(cos_g < 0.7) & (r1 > 0) & (rbr_raw < -0.01), 2].astype(np.float32));
                err = r1 - r0;
                err -= np.nanmean(err)
                dif = st.rolling_mean2(abs(err), 100)
                err = st.rolling_mean2(err, 5)
                dif2 = maximum_filter(np.abs(err), 5)

            sky = (rbr < 126) & (dif < 1.2);
            sky |= dif < 0.9;
            sky |= (dif < 1.5) & (err < 3) & (rbr < 105)
            sky |= (rbr < 70);
            sky &= (self.red > 0);
            cld = (dif > 2) & (err > 4);
            cld |= (self.red > 150) & (rbr > 160) & (dif > 3);
            # Clouds with high rbr
            cld |= (rbr > 180);
            cld[cos_g > 0.7] |= (rgb_curr[cos_g > 0.7, 2] < mblue) & (
                    rbr_raw[cos_g > 0.7] > -0.01);  # dark clouds
            cld &= dif > 3
            total_pixel = np.sum(r1 > 0)

            min_size = 50 * self.camera.nx / 1000
            cld = remove_small_objects(cld, min_size=min_size, connectivity=4, in_place=True)
            sky = remove_small_objects(sky, min_size=min_size, connectivity=4, in_place=True)

            ncld = np.sum(cld);
            nsky = np.sum(sky)

            # These thresholds don't strictly need to match those used in forecasting / training
            if (ncld + nsky) <= 1e-2 * total_pixel:
                logging.info("\tNo clouds")
                return False
            # Shortcut for clear or totally overcast conditions
            elif (ncld < nsky) and (ncld <= 5e-2 * total_pixel):
                self.cloud_mask = cld.astype(np.uint8)
                logging.info(self.filename + ": clear")
                # self.layers = 1
                return False
            #elif (ncld > nsky) and (nsky <= 5e-2 * total_pixel):
            #    self.cloud_mask = ((~sky) & (r1 > 0)).astype(np.uint8)
            #    logging.info(self.filename + ": overcast")
            #    # self.layers = 1
            #    return

            max_score = -np.Inf
            x0 = -0.15;
            ncld = 0.25 * nsky + 0.75 * ncld
            nsky = 0.25 * ncld + 0.75 * nsky
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # The logic of the following loop is questionable. The cloud_mask can be defined and overwritten
                # at each iteration or "not at all" if the last condition "score > max_score" is never satisfied
                for slp in [0.1, 0.15]:
                    offset = np.zeros_like(r1);
                    mk = cos_g < x0;
                    offset[mk] = (x0 - cos_g[mk]) * 0.05;
                    mk = (cos_g >= x0) & (cos_g < 0.72);
                    offset[mk] = (cos_g[mk] - x0) * slp
                    mk = (cos_g >= 0.72);
                    offset[mk] = slp * (0.72 - x0) + (cos_g[mk] - 0.72) * slp / 3;
                    rbr2 = rbr_raw - offset;
                    minr, maxr = st.lower_upper(rbr2[rbr2 > -1], 0.01)
                    rbr2 -= minr;
                    rbr2 /= (maxr - minr);

                    lower, upper, step = -0.1, 1.11, 0.2
                    max_score_local = -np.Inf

                    for iter in range(3):
                        for thresh in np.arange(lower, upper, step):
                            mk_cld = (rbr2 > thresh)  # & (dif>1) & (rbr>70)
                            mk_sky = (rbr2 <= thresh) & (r1 > 0)
                            bnd = st.get_border(mk_cld, 10, thresh=0.2, ignore=self.red <= 0)
                            
                            sc = [np.sum(mk_cld & cld) / ncld, np.sum(mk_sky & sky) / nsky, np.sum(dif2[bnd] > 4) / np.sum(bnd), \
                                -5 * np.sum(mk_cld & sky) / nsky, -5 * np.sum(mk_sky & cld) / ncld,
                                -5 * np.sum(dif2[bnd] < 2) / np.sum(bnd)]
                            score = np.nansum(sc)
                            if score > max_score_local:
                                max_score_local = score
                                thresh_ref = thresh
                                if score > max_score:
                                    max_score = score
                                    # Set the cloud mask
                                    self.cloud_mask = mk_cld.astype(np.uint8);
                                    logging.info(self.filename + ": clouds! " + str(iter) + "," + str(thresh))

                        lower, upper = thresh_ref - 0.5 * step, thresh_ref + 0.5 * step + 0.001
                        step /= 4;
        except Exception as e:
            raise e
            logging.error("Error creating cloud mask: " + str(e))
            return False
        return True
        
    """
        Computes the cloud motion 
    """
    @ray.method(num_returns=1)
    def compute_cloud_motion(self, img_prev, ratio=0.7, threads=1):

        if (img_prev is None):
            logging.info("\tCannot compute cloud motion without previous image")
            return False
        if (img_prev.rgbu is None or img_prev.red is None):
            logging.info("\tCannot compute cloud motion without previous image")
            return False        

        if (self.cloud_mask is None):
            if (img_prev.rgbu is None): #if prev image doesn't exist, this is expected
                logging.info("\tCannot compute cloud motion on images where the cloud mask has not been defined")
            else:
                logging.warning("\tCannot compute cloud motion on images where the cloud mask has not been defined")
            return False

        if (self.bright_mask is None):
            if (img_prev.rgbu is None): #if prev image doesn't exist, this is expected
                logging.info("\tCannot compute cloud motion on images where the cloud mask has not been defined")
            else:
                logging.warning("\tCannot compute cloud motion on images where the bright mask has not been defined")
            return False

        # Return if there are no clouds
        if np.sum(self.cloud_mask > 0) < (2e-2 * self.camera.nx * self.camera.ny):
            logging.info("\tCloud free case")
            return False

        r0 = img_prev.red.astype(np.float32)
        r0[r0 <= 0] = np.nan
        r1 = self.red.astype(np.float32)
        r1[r1 <= 0] = np.nan

        semi_static = self.bright_mask == 1
        r1[semi_static] = np.nan

        ny, nx = r1.shape

        try:
            mask0 = r0 > 0
            mask1 = morphology.binary_dilation(self.cloud_mask, np.ones((15, 15)));
            mask1 &= (r1 > 0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)     #expect RuntimeWarnings here
                corr = mncc.mncc(r0, r1, mask1=mask0, mask2=mask1, ratio_thresh=ratio, threads=threads)

            if np.count_nonzero(~np.isnan(corr)) == 0:
                logging.info("\tNaN slice encountered")
                return False

            max_idx = np.nanargmax(corr)
            vy = max_idx // len(corr) - ny + 1
            vx = max_idx % len(corr)  - nx + 1

            if np.isnan(vy):
                logging.info("\tThe cloud motion velocity is NaN")
            else:
                self.velocity += [[vy, vx]]
        except:
            logging.error("\tAn error occurred computing the cloud motion")
            return False
            
        return True

#############################################################################

# Utils

"""
    Get the image NetCDF filename associated to the timestamp of the JPEG image
"""

def get_ncdf_curr_filename(filename_jpeg, tmpfs):

    try:
        basename = os.path.splitext(os.path.basename(filename_jpeg))[0]
        btimes = basename[-14:-6]
        filename_curr = tmpfs + btimes + '/' + basename + '.nc'

        return filename_curr
    except TypeError:
        return None

"""
    Get the image NetCDF filename associated to the timestamp of the JPEG image - 30 seconds
"""

def get_ncdf_prev_filename(filename_jpeg, tmpfs, timezone):
    
    try:
        basename_curr = os.path.splitext(os.path.basename(filename_jpeg))[0]
        btimes_curr = basename_curr[-14:]
        t_curr = utils.localToUTC(datetime.strptime(btimes_curr, '%Y%m%d%H%M%S'), timezone)
        t_prev = t_curr - timedelta(seconds=30)
        btimes_prev = t_prev.strftime('%Y%m%d%H%M%S')

        basename_prev = basename_curr.replace(btimes_curr, btimes_prev);
        btimes = basename_prev[-14:-6]
        filename_prev = tmpfs + btimes + '/' + basename_prev + '.nc'

        return filename_prev
    except TypeError:
        return None
    
def get_prev_filename(filename_jpeg):

    try:
        basename_curr = os.path.splitext(os.path.basename(filename_jpeg))[0]
        btimes_curr = basename_curr[-14:]
        t_curr = datetime.strptime(btimes_curr, '%Y%m%d%H%M%S')
        t_prev = t_curr - timedelta(seconds=30)
        btimes_prev = t_prev.strftime('%Y%m%d%H%M%S')

        filename_prev = filename_jpeg.replace(btimes_curr, btimes_prev);

        return filename_prev
    except TypeError:
        return None

"""
    Restores an image object from a NetCDF file
"""
def restore_ncdf(camera, filename, memory=None):

    logging.info("\tReading image file " + filename)
    try:

        root_grp = nc4.Dataset(filename, 'r', memory=memory)  # 'r' stands for read

        if 'file_source' in root_grp.ncattrs():
            file_source = root_grp.getncattr('file_source')
        else:
            file_source = ''

        img = Image(file_source, camera)

        if 'DayTime' in root_grp.variables:
            img.day_time = root_grp.variables['DayTime'][0]
        if 'SolarAltitudeAngle' in root_grp.variables:
            img.saa = root_grp.variables['SolarAltitudeAngle'][0]
        if 'SolarAzimuthAngle' in root_grp.variables:
            img.saz = root_grp.variables['SolarAzimuthAngle'][0]
        if 'RGBu' in root_grp.variables:
            img.rgbu = root_grp.variables['RGBu'][:]
        if 'BrightMask' in root_grp.variables:
            img.bright_mask = root_grp.variables['BrightMask'][:]
        if 'Red' in root_grp.variables:
            img.red = root_grp.variables['Red'][:]
        if 'CloudMask' in root_grp.variables:
            img.cloud_mask = root_grp.variables['CloudMask'][:]
        if 'CloudMotion' in root_grp.variables:
            img.velocity = root_grp.variables['CloudMotion'][:].tolist()
        if 'CloudBaseHeight' in root_grp.variables:
            img.cloud_base_height = root_grp.variables['CloudBaseHeight'][:].tolist()
            img.height_neighbours = list(root_grp.variables['CloudBaseHeight'].getncattr('neighbours').split(','))

        root_grp.close()

    except Exception as e:
        #raise e
        logging.error("\tAn error occurred reading the NetCDF file " + filename + ": " + str(e))
        return None

    return img

#############################################################################