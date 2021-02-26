import numpy as np
import stat_tools as st
from datetime import datetime,timedelta
import os
from skimage.morphology import remove_small_objects
from scipy.ndimage import morphology
import mncc, geo
import glob, pickle
from collections import deque
import pytz
import utils

#############################################################################
# Module constants

#BND_WIN = 20;
#BND_RED_THRESH, BND_RBR_THRESH  = 2/1.5, 0.012/2
#DRED_THRESH, DRBR_THRESH = 150, 157
#STD_RED_THRESH, STD_RBR_THRESH = 1.2, 0.012

# Deprecated (use numpy.deg2rad instead)
#deg2rad=np.pi/180

#############################################################################
# The camera object

# TODO define methods save and restore
# TODO define method geolocation
class camera:

    # Variable with the suffix '0' means it is for the raw, undistorted image
    def __init__(self, camID, lat, lon, nx0, ny0, xstart, ystart, rotation, beta, azm, c1, c2, c3, \
                 max_theta=70, nxy=2000, timezone=pytz.timezone("UTC"), fpath=None, overwrite=False, ignore_mask=False):

        # Size of the undistorted image, note that the input parameter ny is never used
        if nxy >= 2000:
            nxy = 2000
        else:
            nxy = 1000

        self.camID = camID
        self.nx = nxy
        self.ny = nxy

        if not overwrite:
            # Check if the camera object has already been precomputed
            try:
                ifile = fpath + self.camID + '.' + str(self.nx) + '.pkl'
                with open(ifile, 'rb') as input:
                    print("Loading precomputed camera " +ifile)
                    self.__dict__ = pickle.load(input).__dict__
                    self.timezone = timezone  # overwrite timezone! :?
                return
            except:
                pass;


        self.lat = lat
        self.lon = lon
        self.nx0 = int(nx0 + 0.5)
        self.ny0 = int(ny0 + 0.5)
        self.xstart = int(xstart - nx0 / 2 + 0.5)
        self.ystart = int(ystart - ny0 / 2 + 0.5)
        # The rotation angle
        self.rotation = rotation
        self.beta = beta
        self.azm = azm
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.max_theta = max_theta
        self.timezone = timezone
        self.path = fpath

        # Build the index tuples of the region of interest
        self.roi = np.s_[self.ystart:self.ystart + self.ny0, self.xstart:self.xstart + self.nx0]

        # Computed below
        self.theta0 = None
        self.phi0 = None
        self.valid = None
        self.cos_p = None
        self.sin_p = None
        self.max_tan = None
        self.weights = None

        # Compute the zenith and azimuth angles for each pixel
        # // = floor division
        # Note that the "linspace" misses 1 element if we want the arrays to be spaced by 1
        # np.linspace(-2,2,4) != np.linspace(-1.5,1.5,4) != np.linspace(-2,2,5)
        x0, y0 = np.meshgrid(np.linspace(-self.nx0 // 2, self.nx0 // 2, self.nx0),
                             np.linspace(-self.ny0 // 2, self.ny0 // 2, self.ny0))
        nr0 = (nx0 + ny0) / 4
        r0 = np.sqrt(x0 ** 2 + y0 ** 2) / nr0

        roots = np.zeros(51)
        rr = np.arange(51) / 100.0
        for i, ref in enumerate(rr):
            # The last root (P(x)=0) of the polynomial P with coefficients c3,c2,c1,ref
            # p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
            roots[i] = np.real(np.roots([self.c3, 0, self.c2, 0, self.c1, -ref])[-1])
        theta0 = np.interp(r0 / 2, rr, roots)

        # phi (i.e.,azimuth) is reckoned with -pi corresponding to north, increasing clockwise
        # NOTE: pysolar use sub-standard definition
        phi0 = np.arctan2(x0, y0) - self.rotation
        phi0 = phi0 % (2 * np.pi)

        # Correction for the mis-pointing error
        k = np.array((np.sin(self.azm), np.cos(self.azm), 0))
        a = np.array([np.sin(theta0) * np.cos(phi0), np.sin(theta0) * np.sin(phi0), np.cos(theta0)]);
        a = np.transpose(a, [1, 2, 0])
        b = np.cos(self.beta) * a + np.sin(self.beta) * np.cross(k, a, axisb=2) \
            + np.reshape(np.outer(np.dot(a, k), k), (self.ny0, self.nx0, 3)) * (1 - np.cos(self.beta))
        self.theta0 = np.arctan(np.sqrt(b[:, :, 0] ** 2 + b[:, :, 1] ** 2) / b[:, :, 2])
        self.phi0 = np.arctan2(b[:, :, 1], b[:, :, 0]) % (2 * np.pi)

        max_theta = np.deg2rad(max_theta)
        # max_theta *= deg2rad
        valid0 = (theta0 < max_theta) & (theta0 > 0);

        max_tan = np.tan(max_theta)
        xbin = np.linspace(-max_tan, max_tan, self.nx)
        ybin = np.linspace(-max_tan, max_tan, self.ny)

        # (xgrid, ygrid) are the grids of the undistorted space
        xgrid, ygrid = np.meshgrid(xbin, ybin)
        rgrid = xgrid * xgrid + ygrid * ygrid
        self.valid = rgrid <= max_tan * max_tan
        self.cos_th = 1 / np.sqrt(1 + rgrid)
        rgrid = np.sqrt(rgrid)
        self.cos_p = ygrid / rgrid
        self.sin_p = xgrid / rgrid
        self.max_tan = max_tan

        x = theta0 + np.nan
        y = theta0 + np.nan
        r = np.tan(theta0[valid0])
        x[valid0] = r * np.sin(phi0[valid0])
        y[valid0] = r * np.cos(phi0[valid0])

        if ignore_mask:
            print("\tIgnoring static mask")
        else:
            try:
                invalid = np.load(fpath + self.camID + '_mask.npy')
                if (self.nx,self.ny) == invalid.shape:
                    self.valid &= ~invalid.astype(np.bool)
                elif (self.nx <= invalid.shape[0]):
                    tmp = st.block_average2(invalid.astype(np.float32), 2)
                    self.valid &= (tmp[:,:,0] < 0.2)
            except FileNotFoundError:
                print("Warning: No static mask found, cannot be applied")
            except Exception as e:
                print("WError loading static mask, cannot be applied: %s" % str(e))

        self.weights = st.prepare_bin_average2(x, y, xbin, ybin);

        ofile = fpath + camID + '.' + str(self.nx) + '.pkl'

        with open(ofile, 'wb') as output:  # Overwrites any existing file.
            print("Saving camera file "+ofile)
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

#############################################################################

def cloud_motion_fft(convolver,fft1,fft2,ratio=0.75):
    """
    Determine cloud motion using existing convolver and fft objects
    Input: Convolver, and two fft objects 
    Output: Cloud motion vector, and max correlation
    """    
####use this routine if convolver and fft objects are ready    
    ny,nx=fft2[-2]
    try:
        corr=mncc.mncc_fft(convolver, fft1, fft2, ratio_thresh=ratio) 
#        plt.figure(); plt.imshow(corr); plt.show()    
        max_idx=np.nanargmax(corr)
        vy,vx=max_idx//len(corr)-ny+1,max_idx%len(corr)-nx+1    
        return vy,vx,corr.ravel()[max_idx] 
    except:
        return np.nan, np.nan, np.nan
            
def cloud_motion_fast(im1,im2,mask1=None, mask2=None,ratio=0.7, threads=1):
    """
    Determine cloud motion 
    Input: Images and masks for two frames
    Output: Cloud motion vector, and max correlation
    """        
####use this routine if the inputs are raw images   
    ny,nx=im2.shape 
#     if im1.dtype == np.uint8:
#         im1 = im1.astype(np.float32);
#     if im2.dtype == np.uint8:
#         im2 = im2.astype(np.float32);
    try:
        corr=mncc.mncc(im1,im2,mask1=mask1,mask2=mask2,ratio_thresh=ratio,threads=threads)     
#         plt.figure(); plt.imshow(corr); plt.show()    
        max_idx=np.nanargmax(corr)
        vy,vx=max_idx//len(corr)-ny+1,max_idx%len(corr)-nx+1    
        return vy,vx,corr.ravel()[max_idx]
    except:
        return np.nan, np.nan, np.nan

def cloud_height(img1,img2,layer=0,distance=None):
    """
    Determine the cloud height for each cloud layer in img1
    Input: Two image objects 
    Output: Cloud height, and max correlation
    """
    if img1.layers<=0 or layer<=0:
        return []
        
    if img1.max_theta != img2.max_theta:
        print("The max_theta of the two cameras is different.");
        return np.nan, np.nan;
    if distance is None:
        distance = 6367e3*geo.distance_sphere(img1.lat,img1.lon,img2.lat,img2.lon)  

#   max_tan=np.tan(img1.max_theta*deg2rad)
    max_tan=np.tan(np.deg2rad(img1.max_theta))

    im1=img1.red.astype(np.float32); im2=img2.red.astype(np.float32)
#     im1=img1.rgb[:,:,0].astype(np.float32); im2=img2.rgb[:,:,0].astype(np.float32)

    
#     mask_tmpl=(img1.cm==layer) 
    mask_tmpl=(img1.cm==1) if layer==1 else (~(img1.cm==1) & (im1>0))           
        
    res = np.nan;
    try:
        corr=mncc.mncc(im2,im1,mask1=im2>0,mask2=mask_tmpl,ratio_thresh=0.5)       
        if np.any(corr>0):
            max_idx=np.nanargmax(corr)
            deltay,deltax=max_idx//len(corr)-img2.ny+1,max_idx%len(corr)-img2.nx+1            
            deltar=np.sqrt(deltax**2+deltay**2)            
            height=distance/deltar*img1.nx/(2*max_tan)
            score=st.shift_2d(im1,deltax,deltay); score[score<=0]=np.nan; 
            score-=im2; score=np.nanmean(np.abs(score[(im2>0)]))
            score0=np.abs(im2-im1); score0=np.nanmean(score0[(im2>0) & (im1>0)])
#             print('Height',img1.camID,img2.camID,deltay,deltax,height,score0,score)
#             fig,ax=plt.subplots(1,2,sharex=True,sharey=True);  ax[0].set_title(str(deltax)+','+str(deltay));
#             ax[0].imshow(im2); ax[1].imshow(im1); plt.show();            
            if score0-score<=0.3*score0:
                res=np.nan
            else:
                res = min(13000,height)

    except:
        print('Cannot determine cloud height.');
#     print(np.nanmax(corr),height,deltay, deltax)
    return res             
         

"""
    Creates the image object, undistorts and computes the cloud mask
"""
def preprocess(cam,filename,outpath):

    if not os.path.isdir(outpath+filename[-18:-10]):
        os.makedirs(outpath+filename[-18:-10])

    t=utils.localToUTC(datetime.strptime(filename[-18:-4],'%Y%m%d%H%M%S'),cam.timezone);
    t_prev=t-timedelta(seconds=30);
    t_prev=t_prev.strftime('%Y%m%d%H%M%S');
    filename_prev=filename.replace(filename[-18:-4],t_prev);

    # Only process the image if the cloud mask can be computed using the previous image
    if len(glob.glob(filename_prev))<=0:
        return None

    print("\tpreprocess->fn=%s\n\t\tt=%s\t\tt_prev=%s\n" % (filename, str(t), str(t_prev)))
    flist=[filename_prev,filename]
    q=deque();

    for f in flist:
        # Create the object image (important fields: rgb, red, rbr, and cm)
        img_curr=image(f,cam);

        # Undistortion
        img_curr.undistort(cam,rgb=True);

        if img_curr.rgb is None:
            return None

        q.append(img_curr)

        if len(q)<=1: 
            continue
        # len(q) is always 2 beyond this point

        img_prev = q[-2];

        r1=img_prev.red.astype(np.float32);
        r1[r1<=0]=np.nan;
        r2=img_curr.red.astype(np.float32);
        r2[r2<=0]=np.nan;
        err0 = r2-r1;
       
        dif=np.abs(err0); 
        dif=st.rolling_mean2(dif,20);
        semi_static=(abs(dif)<10) & (r1-127>100);
        semi_static=morphology.binary_closing(semi_static,np.ones((10,10)));
        semi_static=remove_small_objects(semi_static,min_size=200, in_place=True);

        img_curr.rgb[semi_static]=0;
        r2[semi_static]=np.nan;

        # One-layer cloud masking
        img_curr.cloud_mask(cam,img_prev);

        # If the cloud mask couldn't be define, remove the image from the queue
        # This is unnecessary
        if (img_curr.cm is None):
            q.popleft();             
            continue
        # Cloud free case
        if (np.sum((q[-1].cm>0))<2e-2*img.nx*img.ny):
            img_curr.layers=0;
        else:               
            dilated_cm=morphology.binary_dilation(q[-1].cm,np.ones((15,15))); dilated_cm &= (r2>0)
            vy,vx,max_corr = cloud_motion(r1,r2,mask1=r1>0,mask2=dilated_cm,ratio=0.7,threads=4);
            if np.isnan(vy):  
                img_curr.layers=0;
            else:
                img_curr.v += [[vy,vx]];
                img_curr.layers=1;
       
        img_curr.dump(outpath+f[-18:-10]+'/'+f[-23:-4]+'.pkl');
        q.popleft();             

    return img_curr

class stitch:
    ###image class
    def __init__(self, time):        
        t_local=time
        self.rgb=None
        self.cm=None

    def dump_stitch(self,filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print("writing: %s" % filename)
