from datetime import datetime
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import netCDF4 as nc4
import stat_tools as st
import numpy as np
import mncc
import geo
import os
import pickle
import warnings
#import image_pipeline
import logging
import ray


def catch(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return None

@ray.remote
class Stitch(object):
    def __init__(self, t=None, stitch_path=None):

        self.btime = t
        self.stitch_path = stitch_path

        self.rgb = None
        self.saa = None
        self.saz = None

        self.lat = None
        self.lon = None

        self.pixel_size = None
        self.cloud_mask = None
        self.bright_mask = None

        self.cloud_base_height = None
        self.raw_cloud_heights = []
        self.height_neighbours = []
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
    @ray.method(num_returns=1)
    def save_netcdf(self,filename):

        logging.info("\tSaving stitch file " + filename)
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
                
            if len(self.raw_cloud_heights) > 0:
                hgt = np.asarray(self.raw_cloud_heights)
                root_grp.createDimension('unlimited', None)
                height = root_grp.createVariable('RawCloudHeights', hgt.dtype.str, 'unlimited', zlib=True, complevel=9)
                height.description = "All calculated cloud heights [m]"
                height.neighbours = ",".join(self.height_neighbours)
                height.units = "m"
                height[:] = hgt

            root_grp.close()
            return True

        except:
            logging.error("\tAn error occurred creating the NetCDF file " + filename)
            return False
            
    @ray.method(num_returns=1)
    def generate_stitch(self, img_obs):

        try:
            img_obs = {cid:ray.get(f_obs) for (cid, f_curr), f_obs in img_obs.items()} #simplify key and retrieve objects
            #print("Generating stitch with: " + str(img_obs))
            cam_list = list(img_obs.keys())
            cameras = {cid:obj.camera for cid,obj in img_obs.items()}
            
            overwrite = True    #***** TODO: load from config
            save_fig = True     #***** TODO: load from config
            
            #cam_list, cameras, overwrite, save_fig = args

            stitch_dir = self.stitch_path + self.btime[:8]
            stitch_file = stitch_dir + '/' + self.btime + '.nc'

            if (os.path.isfile(stitch_file)) and (overwrite == False):
                logging.info("\tThe stitch " + stitch_file + " already exists")
                return False
            
            if not os.path.isdir(stitch_dir):
                try:
                    logging.info("Creating directory "+stitch_dir)
                    os.mkdir(stitch_dir)
                except OSError:
                    logging.error("Cannot create directory "+stitch_dir)
                    return False
            
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
            imgs = [i for k, i in img_obs.items()]
            
            #for camID, ncdf in cam_list.items():
            #    imgc = image.restore_ncdf(cameras[camID],ncdf)
            #    if imgc is not None:
            #        imgs += [imgc]

            h = self.raw_cloud_heights
            v = []
            for i, img in enumerate(imgs):
                #if len(img.cloud_base_height) > 0:
                #    h += [img.cloud_base_height]
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

                #self = stitch(self.btime)

                # Solar parameters inherited from the first image
                self.saa = imgs[0].saa
                self.saz = imgs[0].saz

                self.cloud_base_height = height
                self.velocity = v

                pixel_size = 2 * h[ilayer] * max_tan / imgs[0].camera.nx

                #print("Height: ", h, " pixel size: ", pixel_size);

                self.pixel_size = pixel_size

                xlen = 2 * h[ilayer] * max_tan + x_cams
                ylen = 2 * h[ilayer] * max_tan + y_cams

                nstch_x = int(xlen // pixel_size)
                nstch_y = int(ylen // pixel_size)

                # Use the center latitude
                self.lon = lon0 - h[ilayer] * max_tan / deg2km / ((lat0 + lat1)/2.)
                self.lat = lat0 + h[ilayer] * max_tan / deg2km

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

                self.rgb = rgb.astype(np.uint8);
                self.cloud_mask = (msk + 0.5).astype(np.uint8)
                self.bright_mask = (bgh + 0.5).astype(np.uint8)

                #self.save_pickle(self.stitch_path + self.btime[:8] + '/' + self.btime + '.sth');

                # Save NetCDF file
                self.save_netcdf(stitch_file);

                # Save PNG file
                if save_fig:
                    png_file = self.stitch_path + self.btime[:8] + '/' + self.btime + '.png'
                    logging.info("\tSaving png file " + png_file)

                    plt.ioff()  # Turn off interactive plotting for running automatically

                    rgb = self.rgb
                    semi_static = self.bright_mask == 1
                    rgb[semi_static] = 0

                    plt.figure();
                    plt.imshow(rgb, extent=[0, xlen, ylen, 0]);
                    plt.xlabel('East distance, km');
                    plt.ylabel('South distance, km')
                    plt.tight_layout();
                    #plt.show();

                    plt.savefig(png_file);
                    plt.close();
            return True
        except Exception as e:
            logging.warning(self.btime + ": Unable to create stitch, " + str(e))


    """
        Processes cloud heights for the full stitch
    """
    @ray.method(num_returns=1)
    def process_cloud_heights(self, img_obs, height_group):
        r_sum = 0
        img_obs = {cid:f_obs for (cid, f_curr), f_obs in img_obs.items()} #simplify key
        for cid, f_obs in img_obs.items():
            # Compute cloud base height (using pairs of cameras)
            logging.info("Processing cloud height for " + cid)
            #print(self.btime + " img_obs: " + str(img_obs).replace(',',',\n\t  '))
                
            results = [self.compute_cloud_height(ray.get(img_obs[cid]), catch(lambda:ray.get(img_obs[cam_neigh]))) for cam_neigh in height_group[cid]]
            r_sum += sum(x is not None for x in results)
            logging.info("\t  Calculated heights: " + str(r_sum))
        
            #results = ray.get([i.save_netcdf.remote(image_pipeline.get_ncdf_curr_filename(f,tmpfs), inmemory=False) for f, i in zip(base_img_files, base_img_obs)])
            #print("\t  Updated netCDFs: " + str(sum(x is not None for x in results)))  
        
        return r_sum



    """
        Computes the cloud base height for each cloud layer
    """
    @ray.method(num_returns=1)
    def compute_cloud_height(self, base_img, img_neig, layer=0, distance=None):
        #logging.getLogger().setLevel(logging.INFO)
        try:
            if (base_img.cloud_mask is None):
                if (img_neig.cloud_mask is None): #if neigh is also missing cloud mask this is most likely start of time series and this is expected
                    logging.info("\tCannot compute cloud base height on images where the cloud mask has not been defined")
                else:
                    logging.warning("\tCannot compute cloud base height on images where the cloud mask has not been defined")
                return False

            if (base_img.camera.max_theta != img_neig.camera.max_theta):
                logging.warning("\tThe max_theta of the two cameras is different");
                return False
        except AttributeError as e:
            logging.warning("Missing attribute, skipping: " + str(e))
            return False

        if distance is None:
            distance = 6367e3 * geo.distance_sphere(base_img.camera.lat, base_img.camera.lon, img_neig.camera.lat,
                                                    img_neig.camera.lon)

        # Only if lyars > 1
        #if distance > 500:
        #    return

        max_tan = np.tan(np.deg2rad(base_img.camera.max_theta))

        im0 = base_img.red.astype(np.float32)
        im1 = img_neig.red.astype(np.float32)

        mask_tmpl = base_img.cloud_mask == 1
        # mask_tmpl = (base_img.cloud_mask == 1) if layer == 1 else (~(base_img.cloud_mask == 1) & (im0 > 0))

        res = np.nan;
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)    #expect RuntimeWarnings here
                corr = mncc.mncc(im1, im0, mask1=im1 > 0, mask2=mask_tmpl, ratio_thresh=0.5)
                if np.any(corr > 0):
                    max_idx = np.nanargmax(corr)
                    deltay = max_idx // len(corr) - img_neig.camera.ny + 1
                    deltax = max_idx % len(corr) - img_neig.camera.nx + 1
                    deltar = np.sqrt(deltax ** 2 + deltay ** 2)
                    height = distance / deltar * base_img.camera.nx / (2 * max_tan)
                    score = st.shift_2d(im0, deltax, deltay);
                    score[score <= 0] = np.nan;
                    score -= im1;
                    score = np.nanmean(np.abs(score[(im1 > 0)]))
                    score0 = np.abs(im1 - im0);
                    score0 = np.nanmean(score0[(im1 > 0) & (im0 > 0)])

                    if (score0 - score) > (0.3 * score0):
                        res = min(13000, height)
                        if (res < 20 * distance) and (res > 0.5 * distance):
                            self.raw_cloud_heights += [int(res)]
                            self.height_neighbours += [img_neig.camera.camID]
                    else:
                        logging.info(self.btime + ": Low score")
                else:
                    logging.info(self.btime + ": Not enough valid points")
        except Exception as e:
            logging.error(self.btime + ": Cannot determine cloud base height, " +str(e));
            return False
        logging.debug(self.btime + ": " + str((self.height_neighbours, self.raw_cloud_heights)))
        return True #(self.height_neighbours, self.cloud_base_height)
        

    @ray.method(num_return=1)
    def extract_features(self):

        if self.rgb is None:
            print("No stitched image, skipping.")
            continue
        if self.cloud_mask is None:
            print("No cloud mask, skipping.")
            continue

        timestamp = utils.localToUTC(datetime.strptime(f[-17:-3], '%Y%m%d%H%M%S'), cam_tz)
        timestamp = timestamp.timestamp()

        ny, nx = self.cloud_mask.shape
        y, x = (self.lat - GHI_loc[:, 0]) * deg2km, (GHI_loc[:, 1] - self.lon) * deg2km * np.cos(
            np.deg2rad(GHI_loc[0, 0]))

        iys = (0.5 + (y + self.cloud_base_height * np.tan(self.saa) * np.cos(self.saz)) / self.pixel_size).astype(
            np.int32)
        ixs = (0.5 + (x - self.cloud_base_height * np.tan(self.saa) * np.sin(self.saz)) / self.pixel_size).astype(
            np.int32)

        args = [[iGHI, iys[iGHI], ixs[iGHI], ny, nx, self, timestamp, outpath + day[:8], GHI_loc[iGHI], lead_minutes, lead_steps] for iGHI in
                range(len(GHI_loc))]

        # Extract features (list of lists)
        ft = ray.get([extract_MP.remote(arg) for arg in args])
        # Make a flat list out of list of lists [filtering out None elements]
        features = [sublist for sublist in ft if sublist is not None]

        if len(features) == 0:
            print("\tNo features found")
            continue

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
                    cc = colors[idx].reshape(1,-1)  # Make the color a 2D array to avoid value-mapping in case ix, iy length matches the color length (in scatter)
                    ax[0].scatter(ix, iy, s=6, marker='o', c=cc, edgecolors='face')
                    ax[0].text(ix + 25, iy, str(idx_GHI), color=colors[idx], fontsize='x-small',
                               bbox=dict(facecolor='darkgray', edgecolor=colors[idx], boxstyle='round,pad=0'))

        if SAVE_FIG:
            plt.tight_layout();
            plt.savefig(outpath + day[:8] + '/' + f[-18:-4] + '_features.png')
            # plt.show()
            plt.close()











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
    