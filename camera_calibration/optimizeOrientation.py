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
from scipy.optimize import least_squares
import ephem
import configparser as cfg
import yaml
import camcoord
# globals
pi2=np.pi/2.

# Read observed Moon position output from find_moon.py and optimize
# camera parameters by minimizing the mean square angular distance to
# the predicted location from ephem.Moon

# Initial parameters are read from camera_cal_bnl.yaml
#####params: nx0,cy,cx,rotation,beta,azm,c1,c2,c3
# ny0=nx0 are the y and x size of the region of interest 'roi', assumed square
# cy,cx is the central pixel of 'roi'
# rotation is the deviation from North in radian, positive to East.

def meansqdist(x):
    # function to be minimized mean square of angular distance between measured
    # and predicted moon positions
    # vars = string array of parameters to vary in x
    # start with just varying cam.rot
    cam.pr0=x[0]
    cam.cy=x[1]
    cam.cx=x[2]
    cam.rot=x[3]
    cam.beta=x[4]
    cam.azm=x[5]
    cam.c1=x[6]
    cam.c2=x[7]
    cam.c3=(0.5-(cam.c1*pi2+cam.c2*pi2**3))/pi2**5
    cam.azizen()
    ix=(dfref.xmoon+0.5).astype(int)
    iy=(dfref.ymoon+0.5).astype(int)
    otheta=cam.theta0[iy,ix]
    ophi=cam.phi0[iy,ix]
    ptheta=[]
    pphi=[]
    for edate in dfref.ephemDate:
        obs.date=edate
        moon.compute(obs)
        ptheta.append(np.pi/2.-moon.alt)
        pphi.append((moon.az-np.pi)%(2*np.pi))
    ptheta=np.array(ptheta)
    pphi=np.array(pphi)
    return np.mean(camcoord.great_circle_distance(otheta,ophi,ptheta,pphi)**2)

def dist(x):
    # function returns residuals to be
    # minimized by least-squares angular distance between measured
    # and predicted moon positions
    # vars = string array of parameters to vary in x
    # start with just varying cam.rot
    cam.pr0=x[0]
    cam.cy=x[1]
    cam.cx=x[2]
    cam.rot=x[3]
    cam.beta=x[4]
    cam.azm=x[5]
    cam.c1=x[6]
    cam.c2=x[7]
    if constrained_c3:
        cam.c3=(0.5-(cam.c1*pi2+cam.c2*pi2**3))/pi2**5
    else:
        cam.c3=x[8]
    cam.azizen()
    ix=(dfref.xmoon+0.5).astype(int)
    iy=(dfref.ymoon+0.5).astype(int)
    otheta=cam.theta0[iy,ix]
    ophi=cam.phi0[iy,ix]
    try:
        exist=(len(ptheta) == len(otheta))
    except:
        exist=False
    if not exist:
        ptheta=[]
        pphi=[]
        for edate in dfref.ephemDate:
            obs.date=edate
            moon.compute(obs)
            ptheta.append(np.pi/2.-moon.alt)
            pphi.append((moon.az-np.pi)%(2*np.pi))
        ptheta=np.array(ptheta)
        pphi=np.array(pphi)
    return camcoord.great_circle_distance(otheta,ophi,ptheta,pphi)


if __name__ == "__main__":  
    ######load the configuration file
    config_file = sys.argv[1] if len(sys.argv) >= 2 else 'camera_calibration.conf'
    config = cfg.ConfigParser()
    config.read(config_file)

    if len(sys.argv) >= 3:
        cameraIDs = sys.argv[2:] 
    else:
        cameraIDs = eval(config['camera']['cameraIDs'])
    imagepath = config['path']['imagepath']
    outpath = config['path']['outpath']
    moon_obs_ext = config['path']['moon_obs_ext']
    camera_cal_file_list = config['path']['camera_cal_file_list']
    camera_cal_file_optimized = config['path']['camera_cal_file_optimized']
    begin_date=ephem.date(config['optimization']['begin_date'])
    end_date=ephem.date(config['optimization']['end_date'])
    constrained_c3=eval(config['optimization']['constrained_c3'])
    sitelat = float(config['geolocation']['lat'])
    sitelon = float(config['geolocation']['lon'])
    # big outer loop
    for cameraID in cameraIDs:
        # initial camera parameters from camera_cal_file_optimized
        # which should be a copy of camera_cal_file
        cam=camcoord.camera(cameraID,camera_cal_file=camera_cal_file_optimized)
    # calculate azi and zen for each pixel, to optimize vary camera parameters
    # rot,cx,cy,nr0,beta,azm,c1,c2, and optionally c3
        # (if constrained_c3 is False) and recalculate.
        cam.azizen()
        nx0=ny0=cam.nx0
        # pr0 is nx0/2, i.e. initial radius estimate.
        pr0=cam.pr0
        # obs is an ephem observer object
        obs = ephem.Observer();
        # lat, lon are the only parameters in config file specified in deg.
        # ephem and numpy use radian
        obs.lat = np.deg2rad(cam.lat)
        obs.lon = np.deg2rad(cam.lon)
        # moon is an ephem moon object
        moon=ephem.Moon()
    #
    # read moon position data from find_moon_all output .csv files between
    # begin_date and end_date
        moonobsdir=Path(outpath,cam.camID)
        moonobsfiles=sorted(moonobsdir.glob("*"+moon_obs_ext))
        if len(moonobsfiles) < 1:
            print("no moon obs found in {}, run find_moon_all.py first".format(moonobsdir))
            exit(1)
        for moonobsfile in moonobsfiles:
            datestr=moonobsfile.name.split('_')
            d1,d2=datestr[1].split('-')
            d1=ephem.date(datetime.strptime(d1,'%Y%m%d').strftime('%Y-%m-%d'))
            d2=ephem.date(datetime.strptime(d2,'%Y%m%d').strftime('%Y-%m-%d'))
            if d1>end_date or d2<begin_date:
                continue
            # read into pandas DataFrame
            dfrefi=pd.read_csv(moonobsfile,index_col=0)
            # only use good points
            # i.e. roundness check and minimum radius 3 pixels.
            good=((dfrefi.flag & 512) == 0) & (dfrefi.rmoon > 3.) 
            dfrefi=dfrefi[good]
            if len(dfrefi) < 1:
                print("no good data points, skipping "+moonobsfile.name)
                continue
            # now concatenate
            try:
                print('{}: Adding {} records to {}'.format(moonobsfile.name,len(dfrefi),len(dfref)))
                dfref=dfref.append(dfrefi)
            except:
                dfref=dfrefi
    # now minimize mean square distance, second by varying the 3 camera orientation
    # angles
        # constraint optimization pr0,cy,cx should not vary by more than 1%
        ptol=0.99
        # rot should be small -0.5,0.5 (about +-30 deg)
        # beta is the tilt and should be small positive 0,0.2 (11 deg)
        # azm is -pi,pi
        # c1, c2, c3 I am not sure off, but they cannot be 0.
        if constrained_c3:
            x0=np.array([cam.pr0,cam.cy,cam.cx,cam.rot,cam.beta,cam.azm,cam.c1,cam.c2])
            bounds=([cam.pr0*ptol,cam.cy*ptol,cam.cx*ptol,-np.pi,0.,-np.pi,-1.,-1.],
                    [cam.pr0/ptol,cam.cy/ptol,cam.cx/ptol,np.pi,0.2,np.pi,1.,1.])
        else:
            x0=np.array([cam.pr0,cam.cy,cam.cx,cam.rot,cam.beta,cam.azm,cam.c1,cam.c2,cam.c3])
            bounds=([cam.pr0*ptol,cam.cy*ptol,cam.cx*ptol,-np.pi,0.,-np.pi,-1.,-1.,-1.],
                    [cam.pr0/ptol,cam.cy/ptol,cam.cx/ptol,np.pi,0.2,np.pi,1.,1.,1.])
#            
        print('before',x0)
        print('with bounds',bounds)
#        res=minimize(meansqdist,x0,method='BFGS',options={'disp': True})
        res=least_squares(dist,x0,verbose=2,bounds=bounds)
        print('after',res.x)
        if constrained_c3:
        # append the constrained c3
            c1=res.x[6]
            c2=res.x[7]
            res.x=np.append(res.x,(0.5-(c1*pi2+c2*pi2**3))/pi2**5)
        # Note: res.x is type float64, which is not one of the canonical
        # guessable data-types and hence yaml.dump adds additional "!!" tags
        # to fully describe the dumped object, which is ugly.
        # converting to float appears to resolve this make sure that's done in
        # these "*_to_yaml" methods.
        cam.save_dict_to_yaml(res.x[:],camera_cal_file_optimized)
        cam.save_list_to_yaml(res.x[:],camera_cal_file_list)
