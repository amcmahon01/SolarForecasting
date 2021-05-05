import numpy as np
import sys
import tempfile
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

# This program finds all the moon positions in pixel coordinates for the
# images taken during the 3 nights around each full moon
#
# Need to provide an initial estimate of the calibration parameters that 
# params: nx0,cy,cx,rotation,beta,azm,c1,c2,c3
# read in from the camera_cal_bnl.yaml file
# to define the search_window_size px search area around the expected moon
# position
#
# ny0=nx0 are the y and x size of the region of interest 'roi', assumed square
# cy,cx is the central pixel of 'roi'
# rotation is the deviation from North in radian, positive to East.

#ToDo: the interactive features originally used in find_moon.py should be
#      removed from find_moon_all.py because that's impractical. There could
#      instead be a postprocessing step before optimization to eliminate
#      outliers
def on_click(event):
    """Event handler for click inside search box to interactively accept a 
    moon image for processing, visually rejecting images affected by stay 
    light, overexposure, etc. by clicking outside search box"""
    global click_inside_search_box
    on_click_inside_search_box = False
    if event.inaxes:
        ax=event.inaxes
        click_inside_search_box = (event.xdata > ix1) and (event.xdata < ix2) \
          and (event.ydata > iy1) and (event.ydata < iy2)
        plt.close()

def circle(thresh):
    """Return center coord. and mean radius and standard deviation of thresh edge"""
    [yt,xt]=np.nonzero(thresh)
    xc=np.nanmean(xt)
    yc=np.nanmean(yt)
    threshimg=thresh*(thresh*0.+1.)
    diffimg=threshimg-ndimage.binary_erosion(threshimg)
    [yedge,xedge]=np.nonzero(diffimg)
    r=np.sqrt((xedge-xc)**2+(yedge-yc)**2)
    return xc,yc,r.mean(),r.std()
    

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
    outdir = config['path']['outpath']
    moon_obs_ext = config['path']['moon_obs_ext']
    camera_cal_file = config['path']['camera_cal_file']
    diagplts = int(config['find_moon']['diagplts'])
    interactive = eval(config['find_moon']['interactive'])
    search_window_size = int(config['find_moon']['search_window_size'])
    rolling_mean_width = int(config['find_moon']['rolling_mean_width'])
    ghost_size_ratio_min = int(config['find_moon']['ghost_size_ratio_min'])
    thresh_std = float(config['find_moon']['thresh_std'])
    nmoonpixmin = int(config['find_moon']['nmoonpixmin'])
    nmoonpixmax = int(config['find_moon']['nmoonpixmax'])
    nobjmax = int(config['find_moon']['nobjmax'])
    roundness_tolerance = float(config['find_moon']['roundness_tolerance'])
    sitelat = float(config['geolocation']['lat'])
    sitelon = float(config['geolocation']['lon'])
    # big outer loop over cameras
    for cameraID in cameraIDs:
        # initials camera parameters from camera_cal_file
        cam=camcoord.camera(cameraID,camera_cal_file=camera_cal_file)
    # calculate azi and zen for each pixel, to optimize vary camera parameters
    # rot,cx,cy,nr0,beta,azm,c1,c2,c3 and recalculate.
        cam.azizen()
        nx0=ny0=cam.nx0
        # pr0 is nx0/2, i.e. probably initial radius estimate.
        pr0=cam.pr0
        xstart=int(cam.cx-nx0/2+0.5); ystart=int(cam.cy-ny0/2+0.5)
        # now round nx0 to nearest integer for roi dimension
        nx0=int(nx0+0.5); ny0=int(ny0+0.5)
        roi=np.s_[ystart:ystart+ny0,xstart:xstart+nx0]

        # obs is an ephem observer object
        obs = ephem.Observer();
        # lat, lon are the only parameters in config file specified in deg.
        # ephem and numpy use radian
        obs.lat = np.deg2rad(cam.lat)
        obs.lon = np.deg2rad(cam.lon)
        # moon is an ephem moon object
        moon=ephem.Moon()
        sun=ephem.Sun()
        # calculate moon rise and set times for 3 days around each of
        # the last 3 full moons
        fullmoondates=[]
# get image archive date-range
        impath=Path(imagepath,cam.camID)
        imdatelist=sorted(impath.glob('20[12][0-9][01][0-9][0-3][0-9]'))
        imdatefirst=imdatelist[0].name
        imdatelast=imdatelist[-1].name
        obs.date=datetime.strptime(imdatefirst,'%Y%m%d').strftime('%Y-%m-%d')
        imdate=imdatefirst
        while imdate < imdatelast:
            moon.compute(obs)
            full_moon_date=ephem.next_full_moon(obs.date)
            obs.date=full_moon_date
            imdate=obs.date.datetime().strftime('%Y%m%d')
            mrise=obs.previous_rising(moon)
            obs.date=mrise
            mrise=obs.previous_rising(moon)
            obs.date=full_moon_date
            mset=obs.next_setting(moon)
            obs.date=mset
            mset=obs.next_setting(moon)
            fullmoondates.append([mrise,mset])
    #
        for d in fullmoondates:
            mrise,mset=d
            ref=[]      #  [f.name,obs.date,ymean,xmean,rmean,rstd,flag]
            for i in np.arange(mrise,mset):
                imdir=impath.joinpath(ephem.date(i).datetime().strftime('%Y%m%d'))
                flist=sorted(imdir.glob('*.jpg'))
                print(len(flist),'images found in',imdir)
                if len(flist) < 1:
                    continue
                # iterate over each "f", "cnt" is zero based index
                for cnt,f in enumerate(flist):
                    # initialize result vars to 0
                    xmean=ymean=rmean=rstd=0.
                    # use flag for different criteria that weed out images
                    flag=0
                    doy=f.name[-18:-4]
                    obs.date = datetime.strptime(doy,'%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')

                    sun.compute(obs)
                    # skip if Sun is above horizon
                    if sun.alt > 0.:
                        continue
                    moon.compute(obs)
                    # skip if moon is below 0.1 radian
                    if moon.alt < 0.1:
                        continue
                    # zen is the lunar zenith in radian
                    # azi is the lunar azimuth in radian shifted by pi, so South is 0.
                    # ToDo: would be better to leave alone so the 0 to 2*pi discontinuity
                    #       is in the north where the moon never passes, so bilinear
                    #       interpolation to fractional moon pixel location will not
                    #       be a problem

                    zen=np.pi/2-moon.alt
                    azi=(moon.az-np.pi)%(2*np.pi)   
            #
            # calculate approximate moon pixel location to define search region of interest
                    rmoon=2.*zen*pr0/np.pi
                    xmoon=cam.cx+rmoon*np.sin(azi+cam.rot)  # east is on the left side when you look up
                                                # at the sky and +azimuth is east of S.
                    ymoon=cam.cy+rmoon*np.cos(azi+cam.rot)
                    try:
                        iimg=plt.imread(f)
            # create monochrome image
                        img=(0.8*iimg[:,:,2]+0.2*iimg[:,:,0])
                    except:
                        print("Could not read",f)
                        flag+=1
                        ref.append([f.name,obs.date,np.nan,np.nan,np.nan,np.nan,flag])
                        continue
            # check that image dimensions agree with camera paramters ndx0,ndy0
                    ny,nx=img.shape
                    if ny != cam.ndy0 or nx != cam.ndx0:
                        print("image dimensions",nx,ny,"should be",cam.ndx0,cam.ndy0)
                        # Can't continue with wrong image size
                        flag+=2
                        ref.append([f.name,obs.date,np.nan,np.nan,np.nan,np.nan,flag])
                        continue
                    halfwidth=search_window_size//2
                    ix1,ix2=int(max(0,xmoon-halfwidth)),int(min(cam.ndx0,xmoon+halfwidth))
                    iy1,iy2=int(max(0,ymoon-halfwidth)),int(min(cam.ndy0,ymoon+halfwidth))
                    img=img[iy1:iy2,ix1:ix2]
                #     img_m=st.rolling_mean2(img,11)
                #     thresh=img_m>200

    #                img_m=st.rolling_mean2(img,rolling_mean_width,ignore=0)
    #                img_m=img-img_m
    #                img_m-=np.nanmean(img_m)
                    bg=np.nanmedian(img)
                    std=np.nanstd(img)
        # RW 2020-12-28: use half maximum above bg for threshold.
        # RW 2021-01-04: maybe make that a parameter as well, maybe 0.7 of max
                    hm=0.5*(img.max()+bg)
                    threshold=hm
                    # It appears that in cases of large overexposure, e.g. on
                    # the Albany cameras the th bg+thresh_std*std exceeds 255
                    # need to recalculate bg and std without first guess threshold
                    bgimg=img<threshold
                    bg=np.nanmedian(img[bgimg])
                    std=np.nanstd(img[bgimg])
                    hm=0.5*(img.max()+bg)
                    threshold=min(img.max(),max(hm,bg+thresh_std*std))
                    thresh=img>=threshold
                #
                # Need to raise threshold to eliminate more "moon not found"
                    s = ndimage.generate_binary_structure(2,2) # iterate structure, includes diagonal neighbors
                    # ndimage.label finds distinct objects in thresh that are connected
                    # by the iterate structure, cc_num is the number of objects.
                    # If cc_num is > 1 it likely means a corrupted moon image or excess
                    # stray light, let's skip
                    labeled_mask, cc_num = ndimage.label(thresh,s)
    # ToDo: skip the cc_num test and instead use bincount below to test the size
    #       of the labeled objects and skip only if there are multiple sizable ones
                    if cc_num < 1:
                        print("no pix above threshold, skipping",doy)
                        # can't continue
                        flag+=8
                        ref.append([f.name,obs.date,np.nan,np.nan,np.nan,np.nan,flag])
                        continue
                    nobjpix=np.bincount(labeled_mask.flat)
                    labelmoon=nobjpix[1:].argmax()+1
                    nmoonpix=nobjpix[labelmoon]
                    if len(nobjpix)>2:
                        nobjpix[1:].sort()
                        nghostpix=nobjpix[-2]
                    else:
                        nghostpix=0
                    try:
                        thresh = (labeled_mask == labelmoon)
                    except:
                        print("thresh unsuccessful for",doy)
                        # don't know why this would fail, but can't continue
                        # without thresh
                        flag+=16
                        ref.append([f.name,obs.date,np.nan,np.nan,np.nan,np.nan,flag])
                        continue
            # RW 2020-12-26: if this test for number of pixels above threshold
            #                then this should be larger than 9 because moon should
            #                be around 10 or 11 pixels wide, set in config file.
                    if nmoonpix<nmoonpixmin:
                        print(nmoonpix,'moonpix not enough for',doy)
                        # no point in continuing, likely just noise, flag
                        # but continue anyway (or lower nmoonpixmin in config,
                        # to allow for atm. extinction caused underexposure)
                        flag+=32
                        continue
                    # more than nobjmax objects bigger than nmoonpixmin
                    # test that size of second biggest blob is factor of 10 smaller
                    #       than biggest blob.
                    if nghostpix > nmoonpix//ghost_size_ratio_min:
                        print('ghost size',nghostpix,'too large for',doy)
                        # no point in continuing, likely a second image
                        # (e.g. in rain drop), this is working out well
                        flag+=256
                        continue
                    nbigobj=np.count_nonzero(nobjpix[1:]>nmoonpixmin)
                    if nbigobj > nobjmax:
                        print(nbigobj,'objects for',doy)
                        flag+=64
                    # also eliminate too big a blob, likely indicating overexposure or
                    # atmospheric scattering
                    # this by itself is not working too well because of greatly varying
                    # exposure levels.
                    if nmoonpix>nmoonpixmax:
                        print(nmoonpix,'moonpix too large a blob for',doy)
                        flag+=128
    # ToDo: see if ndimage can determine shape, look for and eliminate any asymmetry.                
        # Note: I considered instead of counting each pixel above a threshold,
        #       weigh each pixel by its brightness, e.g.
        #            cm=ndimage.measurements.center_of_mass(img_m,labeled_mask,[1])
        #            ymean=iy1+cm[0][0]
        #            xmean=ix1+cm[0][1]
        #       but the images show a surprisingly large asymmetric brightness variation
        #       which I believe is non-physical but a .jpg artifact, so could instead
        #            cm=ndimage.measurements.center_of_mass(thresh,labeled_mask,[1])
        #            ymean=iy1+cm[0][0]
        #            xmean=ix1+cm[0][1]
        # ToDo: that should really be the same as the steps below, but for some reason
        #       they differ by about 0.2 pixels.
                    # Find coordinates of thresholded image
                    xc,yc,rmean,rstd=circle(thresh)
                    xmean = ix1+xc
                    ymean = iy1+yc
        # this filter seems too simplistic and could break symmetry, if anything it
        # should be based on distance from moon image center
        #            filter=(np.abs(y-np.mean(y))<2.5*np.std(y)) & (np.abs(x-np.mean(x))<2.5*np.std(x))
                    if rstd > rmean*roundness_tolerance:
                        print("circle std",rstd,">",rmean*roundness_tolerance," for",doy)
                        flag+=512
                        ref.append([f.name,obs.date,ymean,xmean,rmean,rstd,flag])
                        continue
                    if ( diagplts > 0 ) and ((cnt % diagplts) == 0):
                        fig,ax=plt.subplots(1,3,sharex=True,sharey=True); 
                        ax[0].imshow(img,extent=([ix1,ix2,iy2,iy1]))
                        ax[0].set_title('raw image')
                        ax[1].imshow(img*thresh,extent=([ix1,ix2,iy2,iy1]))
                        ax[1].set_title('img*thresh')
                        ax[2].imshow(thresh,extent=([ix1,ix2,iy2,iy1]))
                        ax[2].set_title('threshold mask')
                        for ip in range(3):
                            ax[ip].plot(xmean,ymean,'+')
    # moved this interactive filter, after all the others, so only final result
    #       gets checked and approved or rejected.
                    if interactive:
                        fig1,ax1=plt.subplots(figsize=[8.,8.])
    #                    fig1.canvas.mpl_connect('close_event', on_close)
                        ax1.imshow(iimg)
                        ax1.plot(cam.cx,cam.cy,'x')
                        phi=np.deg2rad(np.linspace(0.,360.,361)%360.)
                        xcirc=cam.cx+cam.pr0*np.sin(phi)
                        ycirc=cam.cy+cam.pr0*np.cos(phi)
                        ax1.plot(xcirc,ycirc)          # draw camera FOV circle
                        ax1.plot([ix1,ix2,ix2,ix1,ix1],[iy1,iy1,iy2,iy2,iy1]) # draw search window
                        ax1.plot(xmean,ymean,'+')
                        plt.connect('button_press_event',on_click)
                        plt.show()
                        if click_inside_search_box:
                            # overwrite previous flags
                            flag=0
                        else:
                            print("clicked outside box, reject",doy)
                            flag*=-1
                            flag=min(flag,-1)

                    if xmean>0 and ymean>0:
                        ref.append([f.name,obs.date,ymean,xmean,rmean,rstd,flag])

            # nothing to save, continue
            if len(ref) < 1:
                print("No data for this full moon period: "+mrise.datetime().strftime('%Y%m%d')+'-'+mset.datetime().strftime('%Y%m%d'))
                continue
            # save in .csv
            camoutdir=Path(outdir,cameraID)
            camoutdir.mkdir(exist_ok=True)
            moon_obs_path=Path(camoutdir,cameraID+'_'+mrise.datetime().strftime('%Y%m%d')+'-'+mset.datetime().strftime('%Y%m%d')+moon_obs_ext)
            # If file exists, save in temporary file, make sure we have an
            # outpath file name so we don't go through the whole process and then
            # fail.
    # ToDo: Might be better to save each line incrementally and allow for
    #       safe interruption and restart
            if moon_obs_path.exists():
                outpath=Path(tempfile.NamedTemporaryFile(
                    prefix=moon_obs_path.stem, suffix=moon_obs_path.suffix,
                    delete=False,mode='w+t').name)
                print("Warning ",moon_obs_path.name," exist, trying to create new temporary outpath",outpath.name)
            else:
                outpath=moon_obs_path

            dfref=pd.DataFrame(ref,columns=['image','ephemDate','ymoon','xmoon','rmoon','rstd','flag'])
            dfref=pd.pivot_table(dfref,index='image')
            dfref.to_csv(outpath)
