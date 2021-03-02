from datetime import datetime
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import netCDF4 as nc4
import numpy as np
import os
import pickle
import warnings
import image

class stitch:
    def __init__(self, time):

        # obsolete
        #self.time = time

        self.rgb = None
        self.saa = None
        self.saz = None

        self.lat = None
        self.lon = None

        self.pixel_size = None
        self.cloud_mask = None
        self.bright_mask = None

        self.cloud_base_height = None
        self.velocity = None

    """
        Saves the stitch object in a Pickle file
    """
    def save_pickle(self, filename):

        print("\tSaving stitch file " + filename)
        try:
            with open(filename, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        except:
            print("\tCannot save the file")

    """
        Save the attributes of the stitch object in a NetCDF file
    """
    def save_netcdf(self,filename):

        print("\tSaving stitch file " + filename)
        try:
            root_grp = nc4.Dataset(filename, 'w', format='NETCDF4')  # 'w' stands for write
            root_grp.description = 'SolarForecasting stitch object representation'
            root_grp.history = "Created on " + datetime.utcnow().strftime("%b-%d-%Y %H:%M:%S")

            # Dimensions
            root_grp.createDimension('one', 1)
            root_grp.createDimension('two', 2)

            if self.rgb is not None:
                nx, ny, nz = self.rgb.shape
                root_grp.createDimension('x', nx)
                root_grp.createDimension('y', ny)
                root_grp.createDimension('z', nz)

            if self.saa is not None:
                saa = root_grp.createVariable('SolarAltitudeAngle', 'f4', 'one')
                saa.description = "Solar altitude angle"
                saa[:] = self.saa
            if self.saz is not None:
                saz = root_grp.createVariable('SolarAzimuthAngle', 'f4', 'one')
                saz.description = "Solar azimuth angle"
                saz[:] = self.saz
            if self.rgb is not None:
                rgb = root_grp.createVariable('RGB', self.rgb.dtype.str, ('x', 'y', 'z'), zlib=True, complevel=9)
                rgb.description = "RGB channels"
                rgb[:, :, :] = self.rgb
            if self.lat is not None:
                lat = root_grp.createVariable('Latitude', 'f4', 'one')
                lat.description = "Latitude"
                lat[:] = self.lat
            if self.lon is not None:
                lon = root_grp.createVariable('Longitude', 'f4', 'one')
                lon.description = "Longitude"
                lon[:] = self.lon
            if self.pixel_size is not None:
                psz = root_grp.createVariable('PixelSize', 'f4', 'one')
                psz.description = "Pixel size"
                psz[:] = self.pixel_size
            if self.bright_mask is not None:
                bmsk = root_grp.createVariable('BrightMask', self.bright_mask.dtype.str, ('x', 'y'), zlib=True, complevel=9)
                bmsk.description = "Bright mask"
                bmsk[:, :] = self.bright_mask
            if self.cloud_mask is not None:
                cmsk = root_grp.createVariable('CloudMask', self.cloud_mask.dtype.str, ('x', 'y'), zlib=True, complevel=9)
                cmsk.description = "Cloud mask"
                cmsk[:, :] = self.cloud_mask
            if self.velocity is not None:
                vel = root_grp.createVariable('CloudMotion', 'f4', 'two')
                vel.description = "Cloud motion"
                vel[:] = self.velocity
            if self.cloud_base_height is not None:
                cbh = root_grp.createVariable('CloudBaseHeight', 'f4', 'one')
                cbh.description = "Cloud base height"
                cbh.units = 'km'
                cbh[:] = self.cloud_base_height

            root_grp.close()

        except:
            print("\tAn error occurred creating the NetCDF file " + filename)

#############################################################################

# Wrappers

def generate_stitch(args):

    stitch_path, btimes, cam_list, cameras, overwrite, save_fig = args

    stitch_file = stitch_path + btimes[:8] + '/' + btimes + '.nc'

    if (os.path.isfile(stitch_file)) and (overwrite == False):
        print("\tThe stitch " + stitch_file + " already exists")
        return

    imgs = []

    # Some initialization parameters
    deg2km = np.deg2rad(6367)

    ####################################################################
    # Define lon0 and lat0 from the camera list
    lon, lat = [], []
    for cid in cameras:
        lon += [cameras[cid].lon]
        lat += [cameras[cid].lat]

    lon0, lon1 = np.min(lon), np.max(lon)
    lat0, lat1 = np.max(lat), np.min(lat)

    x_cams = np.abs(lon1 - lon0) * deg2km * np.cos(np.deg2rad(lat1))
    y_cams = np.abs(lat0 - lat1) * deg2km
    ####################################################################

    # Load list of images
    for camID, ncdf in cam_list.items():
        imgc = image.restore_ncdf(cameras[camID],ncdf)
        if imgc is not None:
            imgs += [imgc]

    h = []
    v = []
    for i, img in enumerate(imgs):
        if len(img.cloud_base_height) > 0:
            h += [img.cloud_base_height]
        if len(img.velocity) > 0:
            v += [img.velocity]

    # Clear sky case
    if len(h) <= 0 or len(v) <= 0:
        h = [15]
        v = [[0, 0]]
    else:
        # Do not use np.array because h can have different number of elements i.e. [[9388],[8777,9546]]
        #h = np.nanmedian(np.array(h) / 1e3, axis=0)
        h = np.array([np.nanmedian(np.hstack(h) / 1e3, axis=0)]) # preserve data type
        v = np.nanmedian(np.array(v), axis=0)

    max_tan = np.tan(imgs[0].camera.max_theta * np.pi / 180)
    for ilayer, height in enumerate(h):

        if np.isnan(h[ilayer]):
            continue

        stch = stitch(btimes)

        # Solar parameters inherited from the first image
        stch.saa = imgs[0].saa
        stch.saz = imgs[0].saz

        stch.cloud_base_height = height
        stch.velocity = v

        pixel_size = 2 * h[ilayer] * max_tan / imgs[0].camera.nx

        #print("Height: ", h, " pixel size: ", pixel_size);

        stch.pixel_size = pixel_size

        xlen = 2 * h[ilayer] * max_tan + x_cams
        ylen = 2 * h[ilayer] * max_tan + y_cams

        nstch_x = int(xlen // pixel_size)
        nstch_y = int(ylen // pixel_size)

        # Use the center latitude
        stch.lon = lon0 - h[ilayer] * max_tan / deg2km / ((lat0 + lat1)/2.)
        stch.lat = lat0 + h[ilayer] * max_tan / deg2km

        rgb = np.zeros((nstch_y, nstch_x, 3), dtype=np.float32)
        cnt = np.zeros((nstch_y, nstch_x), dtype=np.uint8)
        msk = np.zeros((nstch_y, nstch_x), dtype=np.float32)
        bgh = np.zeros((nstch_y, nstch_x), dtype=np.float32)

        for i, img in enumerate(imgs):

            # The image is in night time - do not do anything
            if img.day_time == 0:
                continue

            start_x = (img.camera.lon - lon0) * deg2km * np.cos(np.deg2rad(img.camera.lat))
            start_x = int(start_x / pixel_size)
            start_y = (lat0 - img.camera.lat) * deg2km
            start_y = int(start_y / pixel_size)

            tmp = np.flip(img.rgbu, axis=1);  # tmp[img.cm!=ilayer+1,:]=0;
            mk = tmp[..., 0] > 0

            rgb[start_y:start_y + img.camera.ny, start_x:start_x + img.camera.nx][mk] += tmp[mk]
            cnt[start_y:start_y + img.camera.ny, start_x:start_x + img.camera.nx] += mk

            if img.cloud_mask is not None:
                tmp = np.flip(img.cloud_mask, axis=1);  # tmp[img.cm!=ilayer+1,:]=0;
                msk[start_y:start_y + img.camera.ny, start_x:start_x + img.camera.nx][mk] += tmp[mk]

            if img.bright_mask is not None:
                tmp = np.flip(img.bright_mask, axis=1);  # tmp[img.cm!=ilayer+1,:]=0;
                bgh[start_y:start_y + img.camera.ny, start_x:start_x + img.camera.nx][mk] += tmp[mk]

        # TODO the code should take into account division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(3):
                rgb[..., i] /= cnt
            msk /= cnt
            bgh /= cnt

        stch.rgb = rgb.astype(np.uint8);
        stch.cloud_mask = (msk + 0.5).astype(np.uint8)
        stch.bright_mask = (bgh + 0.5).astype(np.uint8)

        #stch.save_pickle(stitch_path + btimes[:8] + '/' + btimes + '.sth');

        # Save NetCDF file
        stch.save_netcdf(stitch_file);

        # Save PNG file
        if save_fig:
            png_file = stitch_path + btimes[:8] + '/' + btimes + '.png'
            print("\tSaving png file " + png_file)

            plt.ioff()  # Turn off interactive plotting for running automatically

            rgb = stch.rgb
            semi_static = stch.bright_mask == 1
            rgb[semi_static] = 0

            plt.figure();
            plt.imshow(rgb, extent=[0, xlen, ylen, 0]);
            plt.xlabel('East distance, km');
            plt.ylabel('South distance, km')
            plt.tight_layout();
            #plt.show();

            plt.savefig(png_file);
            plt.close();

    return

"""
    Restores an stich object from a NetCDF file
"""
def restore_ncdf(filename):

    print("\tReading stitch file " + filename)
    try:

        root_grp = nc4.Dataset(filename, 'r', format='NETCDF4')  # 'r' stands for read

        stch = stitch(filename[-17:-3])

        if 'SolarAltitudeAngle' in root_grp.variables:
            stch.saa = root_grp.variables['SolarAltitudeAngle'][0]
        if 'SolarAzimuthAngle' in root_grp.variables:
            stch.saz = root_grp.variables['SolarAzimuthAngle'][0]
        if 'RGB' in root_grp.variables:
            stch.rgb = root_grp.variables['RGB'][:]
        if 'Latitude' in root_grp.variables:
            stch.lat = root_grp.variables['Latitude'][0]
        if 'Longitude' in root_grp.variables:
            stch.lon = root_grp.variables['Longitude'][0]
        if 'PixelSize' in root_grp.variables:
            stch.pixel_size = root_grp.variables['PixelSize'][0]
        if 'BrightMask' in root_grp.variables:
            stch.bright_mask = root_grp.variables['BrightMask'][:]
        if 'CloudMask' in root_grp.variables:
            stch.cloud_mask = root_grp.variables['CloudMask'][:]
        if 'CloudMotion' in root_grp.variables:
            stch.velocity = root_grp.variables['CloudMotion'][:].tolist()
        if 'CloudBaseHeight' in root_grp.variables:
            stch.cloud_base_height = root_grp.variables['CloudBaseHeight'][0]

        root_grp.close()

    except:
        print("\tAn error occurred reading the NetCDF file " + filename)
        return None

    return stch