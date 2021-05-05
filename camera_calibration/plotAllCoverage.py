import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime
#from scipy import ndimage
from skimage.exposure import equalize_hist
from scipy.optimize import minimize
import ephem
import configparser as cfg
import yaml
import camcoord

# Read camera parameters with camcoord.camera, both original
# (camera_cal_file) and optimized (camera_cal_file_optimized) and moon_obs
# and plot the x,y pixel positions with open circles for bad and closed circles
# for good moon obs.
#####params: nx0,cy,cx,rotation,beta,azm,c1,c2,c3

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
        if plotopt:
            cam=camopt
    # calculate azi and zen for each pixel, to optimize vary camera parameters
    # rot,cx,cy,nr0,beta,azm,c1,c2,c3 and recalculate.
        cam.azizen()
        # obs is an ephem observer object
        obs = ephem.Observer();
    # abort if camopt cam is more than 10 m from opt
    
        # lat, lon are the only parameters in config file specified in deg.
        # ephem and numpy use radian
        obs.lat = np.deg2rad(cam.lat)
        obs.lon = np.deg2rad(cam.lon)
        # moon is an ephem moon object
        moon=ephem.Moon()
        sun=ephem.Sun()
    #
    # read moon position data from find_moon output .csv files
        moonobsdir=Path(outpath,cam.camID)
        moonobsfiles=sorted(moonobsdir.glob("*"+moon_obs_ext))
        fig,ax=plt.subplots()
        # use a dawn image as background, pick a fixed date 2021-02-26 for now.
        obs.date='2021-02-26'
        impath=Path(imagepath,cam.camID)
        imdir=impath.joinpath(obs.date.datetime().strftime('%Y%m%d'))
        # globbing files that match the tens digit of the minute of sunrise.
        # and picking the last of typically 2 files in those 10 min.
        sun.compute(obs)
        globstr=sun.rise_time.datetime().strftime('%Y%m%d%H%M')
        dawnfiles=sorted(imdir.glob(cameraID+"_"+globstr[:-1]+"*.jpg"))
        img=plt.imread(dawnfiles[-1])
        # need to enhance the image or mask or polar plot.
#        stretch e.g. between 20 and 30, or hist_equal?
        imgeq=equalize_hist(img)
        ax.imshow(imgeq)
        ax.plot(cam.cx,cam.cy,'x')
        phi=np.deg2rad(np.linspace(0.,360.,361)%360.)
        xcirc=cam.cx+cam.pr0*np.sin(phi)
        ycirc=cam.cy+cam.pr0*np.cos(phi)
        ax.plot(xcirc,ycirc)          # draw camera FOV circle
        
        d1=np.inf
        d2=-np.inf
        for moonobsfile in moonobsfiles:
        # read into pandas DataFrame
            dfref=pd.read_csv(moonobsfile,index_col=0)
            # Only plot good points for now,
            # i.e. roundness check and minimum radius 3 pixels.
            good=((dfref.flag & 512) == 0) & (dfref.rmoon > 3.) 
            dfref=dfref[good]
            if len(dfref) < 1:
                print("no good data points, skipping "+moonobsfile.name)
                continue
            meandate=np.nanmean(dfref.ephemDate.values)
            mindate=np.nanmin(dfref.ephemDate.values)
            maxdate=np.nanmin(dfref.ephemDate.values)
            ax.plot(dfref.xmoon,dfref.ymoon,'.',label=ephem.date(meandate).datetime().strftime('%Y-%m-%d'))
            id1=dfref.ephemDate.idxmin()
            d1=min(d1,ephem.date(dfref.ephemDate[id1]))
            id2=dfref.ephemDate.idxmax()
            d2=max(d2,ephem.date(dfref.ephemDate[id2]))
        d1=d1.datetime()
        d2=d2.datetime()
        ax.set_title(cameraID+':'+d1.strftime('%Y-%m-%d')+'/'+d2.strftime('%Y-%m-%d'))
        ax.legend(bbox_to_anchor=(1.05,1),loc='upper left',borderaxespad=0.)
        figfile=str(Path(moonobsdir,cameraID+'_allxy_'+d1.strftime('%Y%m%d')+'-'+d2.strftime('%Y%m%d')))
        plt.savefig(figfile)
        plt.show()
        plt.close('all')
