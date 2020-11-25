import numpy as np
import warnings 
from scipy.ndimage.filters import uniform_filter
import matplotlib.pyplot as plt
import rolling

def nans(shape, dtype=np.float32):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def shift_2d(data, dx, dy, constant=np.nan):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param dx: The shift in x
    :param dy: The shift in y
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:np.abs(dx)] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:np.abs(dy), :] = constant
    return shifted_data    

def shift2(data, dx, dy, constant=np.nan):
    return shift_2d(data, dx, dy, constant=np.nan)
    
def bin_average(z, x, xbin, stdout=None,interpolate=False,numout=None):
    n=len(xbin)-1;
    zout=np.zeros(n)+np.nan;
    for i in range(n):
        foo=np.all((x>=xbin[i],x<xbin[i+1]),axis=0);
        if np.sum(foo)>0.5:
            zout[i]=np.nanmean(z[foo]);
            if stdout is not None:
                stdout[i]=np.nanstd(z[foo]);            
            if numout is not None:
                numout[i]=sum(z[foo]>-1e30);
    if interpolate:
        flag=np.isnan(zout);
        coeff=np.polyfit(xbin[~flag], zout[~flag], 1)
        zout[flag]=np.polyval(coeff,xbin[flag]);
    return zout;        

def bin_average_reg(z, x, xbin, interpolate=False,numout=None):
    nx=len(xbin)-1;
    dx=xbin[1]-xbin[0];
    
    ix=(0.5+(x-xbin[0])/dx).astype(int)
    valid=(ix>=0) & (ix<nx) 
    count=np.bincount(ix[valid]);
    s=np.bincount(ix[valid],weights=z[valid])
    zout=np.zeros(nx)+np.nan;
    zout[:len(s)]=(s/count);  
    return zout; 

def bin_average2(z, x, y, xbin, ybin, interpolate=False,numout=None):
    nx=len(xbin); ny=len(ybin);
    dx=np.zeros(nx); dy=np.zeros(ny);
    dx[:nx-1]=0.5*np.diff(xbin); dx[-1]=dx[-2];
    dy[:ny-1]=0.5*np.diff(ybin); dy[-1]=dy[-2];
    zout=np.zeros((ny,nx))+np.nan;
    for j in range(ny):
        for i in range(nx):
            foo=np.all((x>=xbin[i]-dx[max(0,i-1)],x<xbin[i]+dx[i],y>=ybin[j]-dy[max(0,j-1)],y<ybin[j]+dy[j]),axis=0);
            if np.sum(foo)>0.5:
                zout[j,i]=np.nanmean(z[foo]);
                if numout is not None:
                    numout[j,i]=sum(z[foo]>-1e30);
    if interpolate:
       zout=fill_by_interp2(zout,xbin,ybin);
    return zout;  
    
def block_average(y, N=2, idim=0):
    ndim=len(y.shape)
    axs=list(np.arange(ndim))
    axs=axs[idim:]+axs[:idim]
#    y=np.asarray(y,order='F')
    y=np.transpose(y,axs)
    remainder = y.shape[0] % N
    n1 = y.shape[0]-remainder
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result=np.nanmean(np.reshape(y[:n1],(n1/N,N)+y.shape[1:]),1)
    
    if remainder != 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lastAvg = np.nanmean(y[n1:],0)
        result=np.append(result,np.reshape(lastAvg,(1,)+lastAvg.shape),0)
    
    axs=list(np.arange(ndim))
    axs=axs[ndim-idim:]+axs[:ndim-idim]
    result=np.transpose(result,axs)
    return result

def bin_average2_reg(z, x, y, xbin, ybin, fill=False,mask=None):
    nx=len(xbin); ny=len(ybin);
    dx=xbin[1]-xbin[0]; dy=ybin[1]-ybin[0];
    
    ix,iy=(0.5+(x-xbin[0])/dx).astype(int), (0.5+(y-ybin[0])/dy).astype(int)
    valid=(ix>=0) & (ix<nx) & (iy>=0) & (iy<ny)
    ixy=iy*nx+ix;
    count=np.bincount(ixy[valid]);
    s=np.bincount(ixy[valid],weights=z[valid])
    zout=np.zeros((ny,nx))+np.nan;
    zout.ravel()[:len(s)]=(s/count);  
    if fill:
        zout=fill_by_interp2(zout,xbin,ybin, mask=mask);              
    return zout; 

def prepare_bin_average2(x, y, xbin, ybin):
    nx=len(xbin); ny=len(ybin);
    dx=xbin[1]-xbin[0]; dy=ybin[1]-ybin[0];
    
    ix,iy=(0.5+(x-xbin[0])/dx).astype(int), (0.5+(y-ybin[0])/dy).astype(int)
    valid=(ix>=0) & (ix<nx) & (iy>=0) & (iy<ny)
    ixy=iy*nx+ix;
    count=np.bincount(ixy[valid]);
            
    return ny,nx,count,valid,ixy[valid]; 

def fast_bin_average2(z,weights):
    ny,nx,count,valid,ixy=weights
    s=np.bincount(ixy,weights=z[valid])
#     zout=np.zeros((ny,nx).dtype=np.float32)+np.nan;
    zout=np.zeros((ny,nx),dtype=z.dtype);
    zout.ravel()[:len(s)]=(s/count);             
    return zout; 
    
def block_average2(y0, N=2, f=0.02):
#    result=block_average(y,N=N,idim=0);
#    result=block_average(result,N=N,idim=1);
    ny,nx=y0.shape[:2];
    y=y0[:(ny//N)*N,:(nx//N)*N];
    y=y.reshape(ny//N,N,nx//N,N,-1);
    yn=~np.isnan(y);
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning);
        result=np.nanmean(np.nanmean(y,1),2);
        count=np.sum(np.sum(yn,1),2);
        result[count<f*N*N]=np.nan;
    return result    

def rolling_mean(x, N, ignore=None, fill_nan=False): 
    N2=N//2; N3=N-N2-1;
    if ignore is None:
        ignore=np.nan;
    elif ~np.isnan(ignore) and ignore!=0 : 
        x[x==0]=np.inf;
    x=np.insert(x,[0]*(N2+1)+[len(x)]*N3,0)
    flag = (x==ignore) | (np.isnan(x)) | (x==0);    
    x[flag | (x==np.inf)] = 0;    
    cumn = np.cumsum(~flag);
    csum = np.cumsum(x) 
    
    res=(csum[N:]-csum[:-N])/(cumn[N:] - cumn[:-N]) 
    if not fill_nan:
        res[flag[N2+1:-N3]]=ignore;

    return res

def rolling_mean2(x0, N, ignore=None, fill_nan=False):    
    x=x0.astype(np.float32);
    flag=~np.isfinite(x)    
    if ignore is None:
        pass;        
    elif isinstance(ignore,np.ndarray):
        flag |= ignore;        
    elif ignore<np.Inf:
        flag |= (x==ignore);
    x[flag]=0

    res=rolling.rolling_mean2(x,(~flag).astype(np.uint8),N); 
    
#     cumn = (~flag).cumsum(0).cumsum(1);     
#     csum = x.cumsum(0).cumsum(1);  
#     
#     res=np.empty(x.shape)
#     N2=N//2; N3=N-N2-1;   
#     res[:N2+1,:N2+1] = csum[N3:N,N3:N]/(cumn[N3:N,N3:N])
#     res[:N2+1,N2+1:-N3] = (csum[N3:N,N:]-csum[N3:N,:-N])/(cumn[N3:N,N:]-cumn[N3:N,:-N])
#     res[N2+1:-N3,:N2+1] = (csum[N:,N3:N]-csum[:-N,N3:N])/(cumn[N:,N3:N]-cumn[:-N,N3:N])
#     res[N2+1:-N3,N2+1:-N3] = (csum[N:,N:]+csum[:-N,:-N]-csum[N:,:-N]-csum[:-N,N:])/(cumn[N:,N:]+cumn[:-N,:-N]-cumn[N:,:-N]-cumn[:-N,N:])
#     res[-N3:,-N3:]=(csum[-1,-1]+csum[-N:-N2-1,-N:-N2-1]-csum[-1,-N:-N2-1]-csum[-N:-N2-1,-1][:,np.newaxis]) /  \
#                     (cumn[-1,-1]+cumn[-N:-N2-1,-N:-N2-1]-cumn[-1,-N:-N2-1]-cumn[-N:-N2-1,-1][:,np.newaxis])
#     res[-N3:,N2+1:-N3]=(csum[-1,N:]+csum[-N:-N2-1,:-N]-csum[-N:-N2-1,N:]-csum[-1,:-N])/(cumn[-1,N:]+cumn[-N:-N2-1,:-N]-cumn[-N:-N2-1,N:]-cumn[-1,:-N])
#     res[N2+1:-N3,-N3:]=(csum[N:,-1][:,np.newaxis]+csum[:-N,-N:-N2-1]-csum[N:,-N:-N2-1]-csum[:-N,-1][:,np.newaxis])/ \
#                     (cumn[N:,-1][:,np.newaxis]+cumn[:-N,-N:-N2-1]-cumn[N:,-N:-N2-1]-cumn[:-N,-1][:,np.newaxis])
#     res[:N2+1,-N3:]=(csum[N3:N,-1][:,np.newaxis]-csum[N3:N,-N:-N2-1])/(cumn[N3:N,-1][:,np.newaxis]-cumn[N3:N,-N:-N2-1])
#     res[-N3:,:N2+1]=(csum[-1,N3:N]-csum[-N:-N2-1,N3:N])/(cumn[-1,N3:N]-cumn[-N:-N2-1,N3:N])
    
    if  not fill_nan:
        res[flag]=x0[flag];
   
    return res;

# def rolling_mean2(x0, N, ignore=None, mask_ignore=None, fill_nan=False):
#     N2=N//2; N3=N-N2-1;
#     x=x0.copy().astype(np.float32); 
#     if mask_ignore is not None:
#         x[mask_ignore]=np.nan
#     if ignore is None: # or np.isnan(ignore):
#         ignore = np.nan;
#     if ignore!=0: 
#         x[x==0]=np.inf;
#     x = np.insert(np.insert(x, [0]*(N2+1)+[x.shape[1]]*N3, 0,axis=1), [0]*(N2+1)+[x.shape[0]]*N3, 0,axis=0);
#     mask_ignore = (x==ignore) | (np.isnan(x)) | (x==0);
#     x[mask_ignore | (x==np.inf)]=0
# 
#     cumn = ~mask_ignore; cumn = cumn.cumsum(0).cumsum(1);     
#     csum = x.astype(float).cumsum(0).cumsum(1);  
#     
#     cnt=(cumn[N:,N:]-cumn[N:,:-N]-cumn[:-N,N:]+cumn[:-N,:-N]).astype(float); cnt[cnt<=0]=np.nan;
#     res=(csum[N:,N:]-csum[N:,:-N]-csum[:-N,N:]+csum[:-N,:-N]) / cnt; 
# #     plt.figure(); plt.imshow(res)
#     
#     if  not fill_nan:
#         res[mask_ignore[N2+1:-N3,N2+1:-N3]]=ignore;
#    
#     return res;


def rolling_std2(x0, N, ignore=np.nan):
    m=rolling_mean2(x0,N,ignore=ignore);
    m_sq=rolling_mean2(x0**2,N,ignore=ignore);
    return np.sqrt(m_sq-m**2);

def fill_by_mean2(x0, N, mask=None, ignore=np.nan):
    res=rolling_mean2(x0,N,ignore=mask,fill_nan=True)   
#     res=rolling_mean2(x0,N,mask_ignore=mask,fill_nan=True) 
    x0[mask]=res[mask]   
#     plt.figure(); plt.imshow(x0); plt.show();
    return x0;
 
# def fill_by_mean2(x0, N, mask=None, ignore=np.nan):
#     x = x0.copy();
#     if not isinstance(mask, (list,np.ndarray,tuple)):
#         mask=np.isnan(x) if (mask is None) | (mask is np.nan)  else x0==mask;
#     imask=~mask
#     mask_ignore=np.isnan(x) if ignore is np.nan else  (x==ignore);
#     mask_ignore &= imask;    
#     x[mask_ignore | mask | np.isnan(x)]=0; 
#     res=nans(x0.shape);
#     x = np.insert(np.insert(x, 0, 0,axis=1), 0, 0,axis=0);
#    
#     cumn = np.insert(np.insert(imask&(~mask_ignore),0,False, axis=1), 0,False,axis=0); cumn = cumn.cumsum(0).cumsum(1);     
#     csum = x.astype(float).cumsum(0).cumsum(1); 
#     N2=N//2; N3=N-N2-1;
#     cnt=(cumn[N:,N:]-cumn[N:,:-N]-cumn[:-N,N:]+cumn[:-N,:-N]).astype(float); cnt[cnt<=0]=np.nan;
#     res[N2:-N3,N2:-N3]=(csum[N:,N:]-csum[N:,:-N]-csum[:-N,N:]+csum[:-N,:-N]) / cnt;  
#     res[imask]=x0[imask];
#     res[mask_ignore]=ignore;
# #     fig,ax=plt.subplots(1,3,sharex=True,sharey=True); ax[0].imshow(cnt); ax[1].imshow(mask); ax[2].imshow(res) 
#    
#     return res.astype(x0.dtype);
    
def fill_by_interp2(c,xbin=None,ybin=None):
    import scipy.interpolate as itp    
    mask = ~(np.isnan(c))
    if xbin is None:
        xbin=np.arange(c.shape[1]);
    if ybin is None:
        ybin=np.arange(c.shape[0]);        
    xx, yy = np.meshgrid(xbin, ybin)
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    data0 = np.ravel(c[mask] )

#    f = itp.LinearNDInterpolator( xym, data0 ) 
#    return f(np.ravel(xx), np.ravel(yy)).reshape( xx.shape )
    return itp.griddata( xym, data0, (np.ravel(xx), np.ravel(yy)), method='linear' ).reshape( xx.shape )
  
def fill_by_dialate2(cc,nnan=6,iter=1):    
    ny,nx=cc.shape
    for i in range(iter):
        ym,xm=np.nonzero(np.isnan(cc))
        ymi=np.vstack((ym-1,ym-1,ym-1,ym,ym,ym+1,ym+1,ym+1)); xmi=np.vstack((xm-1,xm,xm+1,xm-1,xm+1,xm-1,xm,xm+1))
        xmi[xmi>nx-1]=nx-1; xmi[xmi<0]=0
        ymi[ymi>ny-1]=ny-1; ymi[ymi<0]=0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ti,=np.nonzero(np.nansum(np.isnan(cc[ymi,xmi]),axis=0)<=nnan)
            cc[ym[ti],xm[ti]]=np.nanmean(cc[ymi,xmi],axis=0)[ti]  
    return cc   

def lower_upper(x0,percentile=0.025):
    x=x0.ravel(); x=x[x<np.Inf]    
    hist,bins=np.histogram(x,bins=200)
    bcs=0.5*(bins[:-1]+bins[1:])
    total=np.sum(hist);  cumhist=np.cumsum(hist);
    lower=bcs[np.argmax(cumhist>percentile*total)];
    upper=bcs[200-np.argmax(cumhist[::-1]<=(1-percentile)*total)]; 
    upper=max(lower,upper)
    return lower,upper
  
def get_border(im0, nbuf=3, thresh=0.1,ignore=None):
    im=im0.astype(np.float32); 
    if ignore is not None:
        im[ignore]=1e6
    mn = uniform_filter(im, (nbuf,nbuf)); 
    return (mn<1-thresh) & (mn>thresh); 

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin,ymax+1, xmin,xmax+1

