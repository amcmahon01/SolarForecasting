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
    #
    # read moon position data from find_moon output .csv files
        moonobsdir=Path(outpath,cam.camID)
        moonobsfiles=sorted(moonobsdir.glob("*"+moon_obs_ext))
        res=[]
        resopt=[]
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
            fig,ax=plt.subplots()
            iy=(dfref.ymoon+0.5).astype(int)
            ix=(dfref.xmoon+0.5).astype(int)
            xp=range(len(dfref.index))
            # unfortunately ephem works only with scalar floats so have to loop.
            ptheta=[]
            pphi=[]
            meandate=np.nanmean(dfref.ephemDate.values)
            mindate=np.nanmin(dfref.ephemDate.values)
            maxdate=np.nanmin(dfref.ephemDate.values)
            for edate in dfref.ephemDate:
                obs.date=edate
                moon.compute(obs)
                ptheta.append(np.pi/2.-moon.alt)
                pphi.append((moon.az-np.pi)%(2*np.pi))
            ptheta=np.array(ptheta)
            pphi=np.array(pphi)
            yp=np.rad2deg(camcoord.great_circle_distance(
                cam.theta0[iy,ix],cam.phi0[iy,ix],
                ptheta,pphi))
            ypmean=np.nanmean(yp)
            ypstd=np.nanstd(yp)
            res.append([meandate,mindate,maxdate,ypmean,ypstd])
            ax.plot(xp,yp,'+',label='original mean:{:.3f} std:{:.3f}'.format(ypmean,ypstd))
            ax.plot([xp[0],xp[-1]],[ypmean,ypmean])
            if plotopt:
                ypopt=np.rad2deg(camcoord.great_circle_distance(
                    camopt.theta0[iy,ix],camopt.phi0[iy,ix],
                    ptheta,pphi))
                ypoptmean=np.nanmean(ypopt)
                ypoptstd=np.nanstd(ypopt)
                resopt.append([meandate,mindate,maxdate,ypoptmean,ypoptstd])
                ax.plot(xp,ypopt,'x',label='optimized mean:{:.3f} std:{:.3f}'.format(ypoptmean,ypoptstd))
                ax.plot([xp[0],xp[-1]],[ypoptmean,ypoptmean])
            id1=dfref.ephemDate.idxmin()
            d1=ephem.date(dfref.ephemDate[id1]).datetime()
            id2=dfref.ephemDate.idxmax()
            d2=ephem.date(dfref.ephemDate[id2]).datetime()
            ax.set_title(cameraID+':'+d1.strftime('%Y-%m-%d')+'/'+d2.strftime('%Y-%m-%d'))
            ax.set_xlabel('Observation Number')
            ax.set_ylabel('Angular Distance between Observed and Predicted [Deg]')
            ax.legend()
            # add figure residuals v. ptheta, and pphi
            fig2,ax2=plt.subplots(1,2,sharey=True)
            ax2[0].plot(np.rad2deg(ptheta),yp,'+',
                         label='original mean:{:.3f} std:{:.3f}'.format(ypmean,ypstd))
            ax2[0].plot([np.rad2deg(ptheta.min()),np.rad2deg(ptheta.max())],[ypmean,ypmean])
            if plotopt:
                ax2[0].plot(np.rad2deg(ptheta),ypopt,'x',
                             label='optimized mean:{:.3f} std:{:.3f}'.format(ypoptmean,ypoptstd))
                ax2[0].plot([np.rad2deg(ptheta.min()),np.rad2deg(ptheta.max())],[ypoptmean,ypoptmean])
            ax2[0].set_title(cameraID+':'+d1.strftime('%Y-%m-%d')+'/'+d2.strftime('%Y-%m-%d'))
            ax2[0].set_xlabel('Lunar Zenith Angle [deg]')
            ax2[0].set_ylabel('Angular Distance between Observed and Predicted [Deg]')
            ax2[0].legend()
            pphi=(pphi-np.pi) % (2.*np.pi)
            ax2[1].plot(np.rad2deg(pphi),yp,'+',
                         label='original mean:{:.3f} std:{:.3f}'.format(ypmean,ypstd))
            ax2[1].plot([np.rad2deg(pphi.min()),np.rad2deg(pphi.max())],[ypmean,ypmean])
            if plotopt:
                ax2[1].plot(np.rad2deg(pphi),ypopt,'x',
                             label='optimized mean:{:.3f} std:{:.3f}'.format(ypoptmean,ypoptstd))
                ax2[1].plot([np.rad2deg(pphi.min()),np.rad2deg(pphi.max())],[ypoptmean,ypoptmean])
            ax2[1].set_title(cameraID+':'+d1.strftime('%Y-%m-%d')+'/'+d2.strftime('%Y-%m-%d'))
            ax2[1].set_xlabel('Lunar Azimuth Angle [deg E of N]')
            ax2[1].legend()
            figfile=str(moonobsfile).replace('_moon_obs.csv','.png')
            plt.savefig(figfile)
            plt.close('all')
#            plt.show()
        dfres=pd.DataFrame(res,columns=['ephemDate','minDate','maxDate','resmean','resstd'])
        if plotopt:
            dfresopt=pd.DataFrame(resopt,columns=['ephemDate','minDate','maxDate','resoptmean','resoptstd'])
        fig,ax=plt.subplots()
        d=np.empty(0)
        for edate in dfres.ephemDate.values:
            d=np.append(d,ephem.date(edate).datetime())
        ax.errorbar(d,dfres.resmean,
                        yerr=dfres.resstd,fmt='+',label='original')
        if plotopt:
            d=np.empty(0)
            for edate in dfresopt.ephemDate.values:
                d=np.append(d,ephem.date(edate).datetime())
            ax.errorbar(d,dfresopt.resoptmean,
                        yerr=dfresopt.resoptstd,fmt='x',label='optimized')
        d1=ephem.date(dfres.ephemDate.values[0]).datetime()
        d2=ephem.date(dfres.ephemDate.values[-1]).datetime()
        ax.set_title(cameraID+':'+d1.strftime('%Y-%m-%d')+'/'+d2.strftime('%Y-%m-%d'))
        ax.set_ylabel('Mean Angular Distance between Observed and Predicted [Deg]')
        ax.legend()
        figfile=str(Path(moonobsdir,cameraID+'_all_'+d1.strftime('%Y%m%d')+'-'+d2.strftime('%Y%m%d')))
        plt.savefig(figfile)
        plt.close('all')
