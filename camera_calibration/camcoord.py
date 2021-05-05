# based on camera.py
# this version limited to just doing the mapping of camera pixel coordinates
# to azimuth and zenith angles for optimizing the camera parameters using the
# observed moon locations.
# variable naming convention: n or i are integers, everything else float. 
#
import yaml
import numpy as np
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta,timezone
import os,ephem
from pathlib import Path
import tempfile
from skimage.morphology import remove_small_objects
from scipy.ndimage import morphology,sobel
from scipy.ndimage.filters import maximum_filter,gaussian_filter,laplace,median_filter
import mncc, geo
from scipy import interpolate, stats
import glob,pickle
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pytz

deg2rad=np.pi/180

def localToUTC(t, local_tz):
    """convert local time to UTC"""
    t_local = local_tz.localize(t, is_dst=None)
    t_utc = t_local.astimezone(pytz.utc)
    return t_utc
    
def UTCtimestampTolocal(ts, local_tz):
    """convert UTC time-stamp to local time"""
    t_utc = dt.datetime.fromtimestamp(ts,tz=pytz.timezone("UTC"))
    t_local = t_utc.astimezone(local_tz)
    return t_local

def great_circle_distance(theta1,phi1,theta2,phi2):
    """calculate angular distance between two points on a sphere defined by
    zenith angles theta and azimuth angles phi, all angles in radian."""
    alt1 = np.pi/2.-theta1
    alt2 = np.pi/2.-theta2
    return np.arccos(np.sin(alt1)*np.sin(alt2)+np.cos(alt1)*np.cos(alt2)*np.cos(phi1-phi2))

class camera:
    ###variable with the suffix '0' means it is for the raw, undistorted image
    # init just read in params from camera_cal_bnl.yaml

    def __init__(self, camID, camera_cal_file='camera_cal_bnl.yaml'):
        """Initialize a camera class object"""
        self.camID=camID
        with open(camera_cal_file,"r") as yfile:
            params=yaml.load(yfile)
        # exit gracefully if yfile doesn't open
        self.nx0=params[camID]['nx0']
        self.ny0=self.nx0
    # pr0 is nx0/2, i.e. probably initial radius estimate.
    # pr0 rather than nx0 should be in the camera_cal_SSS.yaml config file
        self.pr0=(self.nx0+self.ny0)/4.
        self.ndy0=params[camID]['ndy0']
        self.ndx0=params[camID]['ndx0']
        self.cx=params[camID]['cx']
        self.cy=params[camID]['cy']
        self.rot=params[camID]['rot']
        self.beta=params[camID]['beta']
        self.azm=params[camID]['azm']
        self.c1=params[camID]['c1']
        self.c2=params[camID]['c2']
        self.c3=params[camID]['c3']
        self.lat=params[camID]['lat']
        self.lon=params[camID]['lon']
# may need to resurrect this
#        xstart=int(params[camID]['cy']-nx0/2+0.5); ystart=int(params[camID]['cx']-ny0/2+0.5)
        self.nx0=int(self.nx0+0.5)
        self.ny0=int(self.ny0+0.5)
        
    def save_dict_to_yaml(self,resxin,outfile):
        """write camera named params into yaml outfile in block style"""
        # first reread original params from outfile
        # lets use the pathlib methods
        outpath=Path(outfile)
        # make sure params are type float and simple list
        # this does not have the desire effect np float is still float64 and
        # float32 is not recognize in np.astype
        resx=list(resxin.astype(float))
        # file should exist
        if outpath.is_file():
            try:
                calparams=yaml.load(outpath.read_text())
            except:
                outpath=Path(tempfile.NamedTemporaryFile(
                    prefix=outpath.stem, suffix=outpath.suffix,
                    delete=False,mode='w+t').name)
                print("Warning ",outfile," could not be read, trying to create new temporary outpath",outpath.name)
                calparams={}
        camID=self.camID
        # doing explicit float here, because the np.astype did not work
        calparams[camID]['nx0']=float(resx[0]*2.)
        calparams[camID]['cx']=float(resx[2])
        calparams[camID]['cy']=float(resx[1])
        calparams[camID]['rot']=float(resx[3])
        calparams[camID]['beta']=float(resx[4])
        calparams[camID]['azm']=float(resx[5])
        calparams[camID]['c1']=float(resx[6])
        calparams[camID]['c2']=float(resx[7])
        calparams[camID]['c3']=float(resx[8])
        calparams[camID]['lat']=self.lat
        calparams[camID]['lon']=self.lon
        outpath.write_text(yaml.dump(calparams,default_flow_style=False))

    def save_list_to_yaml(self,resxin,outfile):
        """write camera params into yaml outfile"""
        outpath=Path(outfile)
        # make sure params are type float and list, resxin must be numpy array
        resx=[]
        for i in range(len(resxin)):
            resx.append(float(resxin[i]))
        if outpath.is_file():
            try:
        # first read outfile to append to
                paramlist=yaml.load(outpath.read_text())
            except:
                outpath=Path(tempfile.NamedTemporaryFile(
                    prefix=outpath.stem, suffix=outpath.suffix,
                    delete=False,mode='w+t').name)
                print("Warning ",outfile," could not be read, trying to create new temporary outpath",outpath.name)
                paramlist={}
        else:
            paramlist={}
        camID=self.camID
        paramlist[camID]=resx[:]
        # first parameter in optimization is pr0 but files expect nx0=2.*pr0
        paramlist[camID][0]*=2.
        outpath.write_text(yaml.dump(paramlist,default_flow_style=False))

    def azizen(self):
        """compute the zenith and azimuth angles for each pixel in image
        and stored in numpy arrays self.theta0, self.phi0"""
        # x0,y0 array pixel coordinates relative to cx,cy
#        ndy0,ndx0=img.shape
        ndy0=self.ndy0
        ndx0=self.ndx0
        x0,y0=np.meshgrid(np.linspace(0,ndx0-1,ndx0)-self.cx,np.linspace(0,ndy0-1,ndy0)-self.cy)
        r0=np.sqrt(x0**2+y0**2)/self.pr0 # fractional radial distance from 0,0
#        self.roi=np.s_[ystart:ystart+self.ny0,xstart:xstart+self.nx0]
        # why not model the zenith angle dependence with polynomial directly
        # rather than linear interpolation between roots.
        roots=np.zeros(51)
        rr=np.arange(51)/100.0
        for i,ref in enumerate(rr):
            roots[i]=np.real(np.roots([self.c3,0,self.c2,0,self.c1,-ref])[-1])
        theta0 = np.interp(r0/2,rr,roots)
                      
        phi0 = np.arctan2(x0,y0) - self.rot  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
        phi0 = phi0%(2*np.pi)

        #####correction for the tilt of the camera
        k=np.array((np.sin(self.azm),np.cos(self.azm),0))
        a=np.array([np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)]); 
        a = np.transpose(a,[1,2,0])
        b=np.cos(self.beta)*a + np.sin(self.beta)*np.cross(k,a,axisb=2) \
          + np.reshape(np.outer(np.dot(a,k),k),(self.ndy0,self.ndx0,3))*(1-np.cos(self.beta))
        theta0=np.arctan(np.sqrt(b[:,:,0]**2+b[:,:,1]**2)/b[:,:,2])
        phi0=np.arctan2(b[:,:,1],b[:,:,0])%(2*np.pi)
#        max_theta *= deg2rad 
#        valid0 = (theta0<max_theta) & (theta0>0); 
#         theta0[valid0]=np.nan;
        self.theta0,self.phi0=theta0,phi0

    def xy(self,theta,phi):
        """method returns raw image pixel coord for given zenith and azimuth 
        angles theta,phi"""
        dist=great_circle_distance(self.theta0,theta,self.phi0,phi)
        [yt,xt]=np.unravel_index(np.argmin(dist),dist.shape)
        return xt,yt
