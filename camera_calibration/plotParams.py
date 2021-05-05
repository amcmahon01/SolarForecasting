import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime
from scipy import ndimage
from scipy.optimize import minimize
import ephem
import configparser as cfg
import yaml
import camcoord

# Read camera parameters with camcoord.camera, both original
# (camera_cal_file) and optimized (camera_cal_file_optimized) and moon_obs
# and plot the angular distance residuals.
#####params: nx0,cy,cx,rotation,beta,azm,c1,c2,c3
def calcroots(c1,c2,c3):                              
     roots=np.zeros(51)           
     rr=np.arange(51)/100.           
     for i,ref in enumerate(rr):   
         roots[i]=np.real(np.roots([c3,0.,c2,0.,c1,-ref])[-1])
         # note np.roots uses polynomial coefficients in reverse order from x**5 to x**0
         # compared to numpy.polynamial module Polynomial, also don't know whether the
         # assumption that the last root "[-1]" is the smallest positive real root is
         # always valid. Would probably be a good idea to instead sort just the real
         # and pick the smallest.
     return rr,roots

if __name__ == "__main__":  
    ######load the configuration file
    config_file = sys.argv[1] if len(sys.argv) >= 2 else 'camera_calibration.conf'
    config = cfg.ConfigParser()
    config.read(config_file)

# ToDo: keyword parameter on command line
# for now assume second argument is moon_obs_filename, parse cameraID from filename

    if len(sys.argv) >= 3:
        cameraIDs = sys.argv[2:] 
    else:
        cameraIDs = eval(config['camera']['cameraIDs'])
    imagepath = config['path']['imagepath']
    outpath = config['path']['outpath']
    moon_obs_ext = config['path']['moon_obs_ext']
    camera_cal_file = config['path']['camera_cal_file']
    camera_cal_file_optimized = config['path']['camera_cal_file_optimized']
    sitelat = float(config['geolocation']['lat'])
    sitelon = float(config['geolocation']['lon'])
    # big outer loop
    for cameraID in cameraIDs:
        # initials camera parameters from camera_cal_file
        cam=camcoord.camera(cameraID,camera_cal_file=camera_cal_file)
        plotopt=True
        try:
            camopt=camcoord.camera(cameraID,camera_cal_file=camera_cal_file_optimized)
        except:
            plotopt=False
    # calculate azi and zen for each pixel, to optimize vary camera parameters
    # rot,cx,cy,nr0,beta,azm,c1,c2,c3 and recalculate.
        cam.azizen()
        if plotopt:
            camopt.azizen()
        # obs is an ephem observer object
        obs = ephem.Observer();
    # abort if camopt cam is more than 10 m from opt
    
        if plotopt and camcoord.great_circle_distance(np.pi/2.-np.deg2rad(cam.lat),
                                              np.deg2rad(cam.lon),
                                              np.pi/2.-np.deg2rad(camopt.lat),
                                              np.deg2rad(camopt.lon)) > (
                                                  10./ephem.earth_radius):
            print("lat lon should agree",cam.lat, cam.lon,camopt.lat,camopt.lon)
            raise RuntimeError("camera moved")
        # lat, lon are the only parameters in config file specified in deg.
        # ephem and numpy use radian
        obs.lat = np.deg2rad(cam.lat)
        obs.lon = np.deg2rad(cam.lon)
        # moon is an ephem moon object
        moon=ephem.Moon()
        fig,axs=plt.subplots(2,1,sharex=True,sharey=True,gridspec_kw={'hspace': 0.0})
        c1=cam.c1
        c2=cam.c2
        c3=cam.c3
        rr0,roots0=calcroots(c1,c2,c3)
        if plotopt:
            rro,rootso=calcroots(camopt.c1,camopt.c2,camopt.c3)
        axs[0].set_title(cameraID+r': $c1=%7.4f$'%c1+r' $c2=%7.4f$'%c2+r' $c3=%7.4f$'%c3)
        for c1p in c1+(-0.01+np.arange(11)*2/1000.):
            # constrain c3 so last root at rr=0.5 is np.pi/2
            x=np.pi/2.
            c3p=(0.5-(c1p*x+c2*x**3))/x**5
            rr,roots=calcroots(c1p,c2,c3p)
            axs[0].plot(2*rr,np.rad2deg(roots-rr*np.pi),'-',label=r"$c1=%7.4f$"%c1p+r" $c3=%7.4f$"%c3p)
        axs[0].plot(2*rr,rr*0,'k-')
        axs[0].plot(2*rr0,np.rad2deg(roots0-rr0*np.pi),'g-',lw=2,label=r"$c1=%7.4f$"%c1+r" $c2=%7.4f$"%c2+r" c3=%7.4f"%c3)
        if plotopt:
            axs[0].plot(2*rro,np.rad2deg(rootso-rro*np.pi),'r-',lw=2,label=r"$c1=%7.4f$"%camopt.c1+r" $c2=%7.4f$"%camopt.c2+r" c3=%7.4f"%camopt.c3)
        axs[0].set_ylabel(r'$\theta - 90r [\deg ]$')
        axs[0].legend()
        
        for c2p in c2+(-0.01+np.arange(11)*2/1000.):
            # constrain c3 so last root at rr=0.5 is np.pi/2
            x=np.pi/2.
            c3p=(0.5-(c1*x+c2p*x**3))/x**5
            rr,roots=calcroots(c1,c2p,c3p)
            axs[1].plot(2*rr,np.rad2deg(roots-rr*np.pi),'-',label=r"$c2=%7.4f$"%c2p+r" $c3=%7.4f$"%c3p)
        axs[1].plot(2*rr,rr*0,'k-')
        axs[1].plot(2*rr0,np.rad2deg(roots0-rr0*np.pi),'g-',lw=2,label=r"$c1=%7.4f$"%c1+r" $c2=%7.4f$"%c2+r" c3=%7.4f"%c3)
        if plotopt:
            axs[1].plot(2*rro,np.rad2deg(rootso-rro*np.pi),'r-',lw=2,label=r"$c1=%7.4f$"%camopt.c1+r" $c2=%7.4f$"%camopt.c2+r" c3=%7.4f"%camopt.c3)
        axs[1].set_xlabel(r'Fractional Distance $r$ from Optical Axis')
        axs[1].set_ylabel(r'$\theta - 90r [\deg ]$')
        axs[1].legend()
        
        plt.show()
