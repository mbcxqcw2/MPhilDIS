import numpy as np
import math as mth
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as col
from matplotlib.colors import LogNorm

def dedisperse(dispersed_image,dispersion_measure,bandwidth,minfreq,sample_time):
    """
    Dedisperses an array containing a pulsar pulse.
    
    Note: the input image must have layout:
    columns -> frequencies (increasing), rows -> time (increasing)
    
    INPUTS:
    
    dispersed_image    : (array-like) a 2D array of frequencies and times to dedisperse
    dispersion_measure : (float) a DM to dedisperse the image at [pc cm^{-3}]
    bandwidth          : (float) the bandwidth of the observation [GHz]
    minfreq            : (float) the minimum frequency observed [GHz]
    sample_time        : (float) the sampling time of the observation [s]
    
    OUTPUTS:
    
    dedispersed_image : (array-like) the dedispersed image
    maxroll           : (int) the number of time samples which have rolled
                              from the end to the beginning of the observation
                              due to array manipulation. These should be cropped
                              before plotting, as they make no physical sense
    """
    
    #set necessary constants
    dedispersed_image=np.zeros_like(dispersed_image,dtype=float)
    const=4.148808*(10**(-3)) #s
    maxfreq=minfreq+bandwidth
    nchannels=np.size(dispersed_image,1)
    #dedisperse image
    for i in range(0,nchannels):
        #dispersion time
        delta_T=const*(((minfreq+(i*bandwidth/nchannels))**(-2))-((maxfreq))**(-2))*dispersion_measure #s
        #convert to number of elements in array to roll back
        #(1 element = sample time, so 1 second = (1/sample time) elements)
        rollback=(delta_T)*(1/((sample_time)))
        #get the maximum rollback for cropping dedispersed array
        if i==0:
            maxroll=int(round(rollback))
        #roll back the elements
        dedispersed_image[:,i]=np.roll(dispersed_image[:,i],-int(round(rollback)))
    
    np.apply_along_axis
    return(dedispersed_image,maxroll)


def crop(dedispersed_image,maxroll):
    """
    Crop time samples from dedispersed image which, after dedispersion,
    rolled from the end of the observation to the beginning.
    
    INPUTS:
    
    dedispersed_image : (array-like) the dedispersed image
    maxroll           : (int) the number of time samples which have rolled
                              from the end to the beginning of the observation
                              due to array manipulation. These should be cropped
                              before plotting, as they make no physical sense.
                              
    OUTPUTS:
    
    cropped_image : (array-like) cropped version of input image.
    
    """
    
    cropped_image=dedispersed_image[maxroll:-maxroll,:]

    return cropped_image

def collapse(cropped_image,sample_time,maxroll):
    """
    Collapses (sums) a cropped image into a 1-dimensional array over the frequency axis.
    Can be used to find the peak signal in the image vs time.
    
    INPUTS:
    
    cropped_image : (array-like) a 2D array of times (axis 0) and frequencies (axis 1) to collapse
    sample_time   : (float) the sampling time of the observation [s]
    maxroll       : (int) the number of time samples which have rolled
                          from the end to the beginning of the observation
                          due to array manipulation. These should be cropped
                          before plotting, as they make no physical sense.
    
    OUTPUTS:
    
    times     : (array-like) times to plot against collapsed observation
    collapsed : (array-like) observation, collapsed over frequency axis
    
    """
    
    #create array of sample times
    times=np.arange(0+maxroll,np.size(cropped_image,0)+maxroll)*sample_time
    
    #create collapsed array of signals by summing over frequencies
    collapsed=np.nansum(cropped_image,axis=1)
    
    return times,collapsed

def Sig_To_Noise(times,data,peakwidthestimate=100):
    """
    Get signal to noise ratio for timeseries data containing a peak.
    Uses an estimate of the peak width in bins, so it can be removed to get a more accurate noise estimate.
    
    INPUTS:
    
    times             : (array-like) times of input data
    data              : (array-like) timeseries data containing a peak
    peakwidthestimate : (int) estimate width of peak in bins
    
    OUTPUTS:
    
    SN      : (float) signal to noise ration of peak
    avnoise : (float) average value of noise in input array
    
    """
    
    #find the peak in the data
    peak=np.amax(data)
    peak_index=np.where(data[:]==peak)
    peak_index=peak_index[0][0]
    peak_time=times[peak_index]
    #remove the signal from the peak
    signal_removed=np.concatenate([data[0:(peak_index-(peakwidthestimate//2))],data[(peak_index+(peakwidthestimate//2)):len(data)]])
    #calculate average noise
    avnoise=np.sum(signal_removed)/len(signal_removed)
    #calculate rms of noise
    rmsnoise=np.sqrt(np.mean(signal_removed**2))
    #calculate S/N ratio
    SN=(peak-avnoise)/rmsnoise
    
    return SN,avnoise

def findpeakinfo(times,data):
    """
    Find a peak in timeseries data.
    
    INPUTS:
    
    times             : (array-like) times of input data
    data              : (array-like) timeseries data containing a peak
    
    OUTPUTS:
    
    peak_time : (float) time of the peak
    peak      : (float) value of the peak
    
    """

    peak=np.amax(data)
    peak_index=np.where(data[:]==peak)
    peak_time=times[peak_index]

    return peak_time,peak

def dDM_step(dispersed_image,timestep,minfreq,bandwidth):
    """
    Get optimum dDM step size for trialling DMs.
    Note: this function takes in GHz and converts to MHz before calculating,
    and also converts the timestep from s to ms
    
    INPUTS:
    
    dispersed_image : (array-like) a 2d array of frequencies and time samples containing a dispersed pulse
    timestep        : (float) the sampling time of the array [seconds]
    minfreq         : the minimum frequency of the observation [Hz]
    bandwidth       : the bandwidth of the observation [Hz]
    
    OUTPUTS:
    
    dDM : the DM step size to use when trialling dispersion measures to find pulse [pc cm^{-3}]
    """
    
    nchannels=float(np.size(dispersed_image,1)) # number of channels in array
                    
    lower_freq=((((minfreq+bandwidth)/(nchannels))*(nchannels/2)))*(10**(-3))     # convert lower frequency to MHz
    upper_freq=((((minfreq+bandwidth)/(nchannels))*((nchannels+1)/2)))*(10**(-3)) # convert upper frequency to MHz
    timestep=timestep*(10**(3))                                                   # convert sampling time to ms
    const = 4.148808*(10**(6))                                                    # from pulsar handbook eq 4.7
    dDM = timestep/(const * ((lower_freq*(10**(-2)))-(upper_freq*(10**(-2)))))    # calculate dDM
    dDM=np.abs(dDM)
    
    return dDM

