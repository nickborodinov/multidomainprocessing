#03/04/2019
import numpy as np
from igor.binarywave import load as loadibw
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from skimage.transform import radon, rescale, resize, downscale_local_mean
from skimage.filters import threshold_otsu, rank
from skimage.morphology import watershed, disk, skeletonize, medial_axis, skeletonize_3d
from skimage.measure import regionprops, find_contours
from tqdm import tqdm
import cv2
from skimage.morphology import watershed
from scipy import ndimage as ndi
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn import decomposition
import pysptools.util as util
import pysptools.eea as eea
import pysptools.abundance_maps as amp
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import pywt
import os
import re
from skimage import data, util
from sklearn.cluster import AgglomerativeClustering,AffinityPropagation,SpectralClustering,DBSCAN,KMeans
from sklearn.metrics import pairwise_distances

def flatten_1d(array,deg=1):
    flattened_array=np.zeros_like(array)
    for i in range(array.shape[0]):
        x=np.arange(0,array.shape[1])
        y=array[i]
        params=np.polyfit(x, y, deg=deg, rcond=None, full=False, w=None, cov=False)
        flattened_array[i]=y-x*params[0]-params[1]
    return flattened_array

def local_fft(array,ham_filter=True):
    h_f=np.ones_like(array)
    if ham_filter:
        size=array.shape
        x = np.arange(0,size[0])
        y = np.arange(0,size[1])
        xx, yy = np.meshgrid(x, y, sparse=True)
        h_f = (np.sin(xx/(size[0]-1)*np.pi)*np.sin(yy/(size[1]-1)*np.pi))**2
    AA=np.fft.fft2(array*h_f, s=None, axes=(-2, -1), norm=None)
    AA=np.fft.fftshift(AA, axes=(-1,-2))
    return AA

def local_radon(array,ham_filter=False):
    h_f=np.ones_like(array)
    if ham_filter:
        size=array.shape
        x = np.arange(0,size[0])
        y = np.arange(0,size[1])
        xx, yy = np.meshgrid(x, y, sparse=True)
        h_f = (np.sin(xx/(size[0]-1)*np.pi)*np.sin(yy/(size[1]-1)*np.pi))**2
    image=array*h_f
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=False)
    return sinogram

def corr2(a,b):
    a = a -  np.sum(a) / np.size(a)
    b = b -  np.sum(b) / np.size(b)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r

def spearman2d(a,b):
    return np.array(scipy.stats.spearmanr(np.ndarray.flatten(a),np.ndarray.flatten(b)))

def find_nearest(array, value):
    array = np.asarray(array)
    if len(array.shape)==1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx=np.array([np.sum(np.abs(array[i]-value)) for i in range(len(array))]).argmin()
    return idx   

def crawldir(topdir=[], ext='sxm'):
    fn = dict()
    for root, dirs, files in os.walk(topdir):
              for name in files:
              
                if len(re.findall('\.'+ext,name)):
                    addname = os.path.join(root,name)

                    if root in fn.keys():
                        fn[root].append(addname)

                    else:
                        fn[root] = [addname]    
    return fn

def norm(array):
    arr=array-np.min(array)
    if np.max(arr)!=0:
        arr=arr/np.max(arr)
    else:
        arr=arr
    return arr 

def radial_profile(data, center,angle_selection='none'):

    if angle_selection=='none':
        selector=np.ones_like(data)
    else:
        array=np.vstack([np.zeros([data.shape[0]//2,data.shape[0]]),np.ones([data.shape[0]//2,data.shape[1]])])
        rotation=angle_selection[0]
        img_rotation_1 = scipy.ndimage.interpolation.rotate(array, angle=rotation,mode ='nearest',reshape =False)*2-1
        array=np.vstack([np.zeros([data.shape[0]//2,data.shape[0]]),np.ones([data.shape[0]//2,data.shape[1]])])
        rotation=180+angle_selection[1]
        img_rotation_2 = scipy.ndimage.interpolation.rotate(array, angle=rotation,mode ='nearest',reshape =False)*2-1
        selector=np.rint(img_rotation_1)*np.rint(img_rotation_2)
        selector=(selector+1)/2
    data=data*selector
    y,x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[0])**2+(y-center[1])**2)
    ind = np.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer
    return radialprofile,selector    

def generate_angular_profile(array,fft_min=10,fft_max=500,step=1,aver=1):
    a=np.zeros([len(list(range(0,180,step)))])
    for i in range(0,180,step):
        b,s=radial_profile(array,center=(array.shape[0]//2,array.shape[1]//2),angle_selection=[i,i+aver])
        b[:fft_min]=0
        b[fft_max:]=0
        a[i]=np.sum(b)
    a=np.roll(a,aver//2)
    return a

def match_histrogram(array_to_match,array):
    coeff=np.std(array)/np.std(array_to_match)
    array_to_match=array_to_match*coeff
    array_to_match=array_to_match-np.mean(array_to_match)+np.mean(array)
    return array_to_match

def chi2_calculator(array1, array2):
    return (np.sum((array1-array2)**2)/(np.ma.size(array1)-1))**0.5

def cross_corr(array_1,array_2):
    spectral_length=array_1.shape[2]
    single_band_cc=np.zeros([spectral_length])
    
    for i in range(spectral_length):
        mean_1_i=np.mean(array_1[:,:,i])
        mean_2_i=np.mean(array_2[:,:,i])
        numerator=np.sum((array_1[:,:,i]-mean_1_i)*(array_2[:,:,i]-mean_2_i))
        denominator=(np.sum((array_1[:,:,i]-mean_1_i)**2)*np.sum((array_2[:,:,i]-mean_2_i)**2))**0.5
        single_band_cc[i]=numerator/denominator
       
    return np.mean(single_band_cc)

def spectral_angle_mapper(array_1,array_2):
    size_x=array_1.shape[0]
    size_y=array_1.shape[1]
    point_array=np.zeros([size_x,size_y])
    
    for i in range(size_x):
        for j in range(size_y):
            numerator=np.sum(array_1[i,j,:]*array_2[i,j,:])
            denominator=np.sum(array_1[i,j,:]**2)**0.5*np.sum(array_2[i,j,:]**2)**0.5
            point_array[i,j]=np.arccos(numerator/denominator)
            
    return np.mean(point_array)
def rmse(array_1,array_2):
    size_x=array_1.shape[0]
    size_y=array_1.shape[1]
    spectral_length=array_1.shape[2]
    
    d2_array_1=np.reshape(array_1,(size_x*size_y,spectral_length))
    d2_array_2=np.reshape(array_2,(size_x*size_y,spectral_length))
    
    return np.linalg.norm(d2_array_1-d2_array_2)/(size_x*size_y*spectral_length)**0.5

def ERGAS(array_1,array_2,downsampling_parameter):
    size_x=array_1.shape[0]
    size_y=array_1.shape[1]
    spectral_length=array_1.shape[2] 
    
    single_band_rmse=np.zeros([spectral_length])
    
    for i in range(spectral_length):
        single_band_rmse[i]=np.linalg.norm(array_1[:,:,i]-array_2[:,:,i])/(size_x*size_y)**0.5
        
        single_band_rmse[i]=single_band_rmse[i]/np.mean(array_1[:,:,i])
        single_band_rmse[i]=single_band_rmse[i]**2
        
    return 100*downsampling_parameter*(np.sum(single_band_rmse)/spectral_length)**0.5