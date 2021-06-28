import pandas as pd
import numpy as np
import os,ndd
import HMM_beads_utils as ut
from embo import EmpiricalBottleneck


def get_windowed_bound(x,z,w,mb=50,nb=2000,p=8):
    '''
    Function that computes the information bottleneck bound between two discrete variables
    Input:
        - x: sequence of past events
        - z: sequence of future events
        - w: size of the window to be applied to x
        - mb: maximum value of the lagrange multiplier beta
        - nb: number of values of beta between 0 and mb to be run through the Blahaut-Arimoto algorithm
        - p: number of cores to use in the computation (to help with speed)
    Output:
        - 4 element tuple with a sequence of ipast and ifuture values corresponding to the information bottleneck for full and one back bounds
    '''
    # Get desired windowed x and corresponding z
    xw,zw = ut.get_windowed_x(x,z,w=w)
    
    # Compute the information bottleneck using EMBO for the window size w specified above
    ipw,ifw,betasw = EmpiricalBottleneck(xw,zw,minsize=True,processes=p,maxbeta=mb,numbeta=nb).get_empirical_bottleneck() 
    
    # Compute the information bottleneck using EMBO for the window size w specified above
    x1b,z1b = ut.get_windowed_x(x,z,w=1)
    ip1b,if1b,betas1b = EmpiricalBottleneck(x1b,z1b,minsize=True,processes=p,maxbeta=mb,numbeta=nb).get_empirical_bottleneck() 
   
    # Return informtion bottleneck
    return((ipw,ifw,ip1b,if1b))

# Test bounds in each session
def get_windowed_mi(x,z,nw):
    '''
    Function that windows of observations from a sequence x offset from y and computes the MI
    Input:
        - x: sequence of "past" observations to be windowed
        - z: sequence that serves as the "future" events
        - nw: maximum window size
    Output:
        - mis: NSB estimated mutual informtion values for each window size
    '''
    mis = np.zeros(nw+1) #vector of mutual informations for different window sizes
    for w in np.arange(1,nw+1):
        xw,zw = ut.get_windowed_x(x,z,w=w)         # use convenience function from utilities to window x
        mis[w] = ut.mutual_inf_nsb(xw,zw,[2**w,2]) # use convenience function from utilities to compute NSB mutual info
    return(mis)

def get_windowed_mi_subs(sdat_all,nw):
    '''
    Function to compute windowed mutual information values for all subjects
    Input:
        - sdat_all: data frame with all of the subject tones, responses, and sources
        - nw: maximum window size to be considered
    Output:
        - Ipast for each window size and Ifuture
    '''
    # Create dictionaries to keep track of variables with subject IDs as the keys
    subs = pd.unique(sdat_all['Subject'])
    ip_sub = {}
    if_sub = {}
    # Loop through each subject and get their Ipast for windows up to nw and their ifuture
    for subi,sub in enumerate(subs):
        sdat = sdat_all[sdat_all['Subject'] == sub]                   # Get data from specific subject 
        ip_sub[sub] = get_windowed_mi(sdat['Bead'],sdat['Resp'],nw)    # compute Ipast using function defined above 
        if_sub[sub] = ut.mutual_inf_nsb(sdat['Resp'],sdat['Jar'],[2,2]) # compute Ifuture
    # Return dictionaries of ipast and ifuture values
    return(ip_sub,if_sub)

def get_windowed_mi_modif(sdat_all,nw):
    '''
    Similar to above, but sdat_all also has relevant history for each entry with nw past beads
    '''
    # Create dictionaries to keep track of variables with subject IDs as the keys
    subs = pd.unique(sdat_all['Subject'])
    ip_sub = {}
    if_sub = {}


def get_bootstrapped_samples(dat,wips,nboot):
    '''
    Function to compute and return bootstrapped distributions of Ipast and Ifuture values for each subject
    Input:
        - dat: all subject data
        - wips: matrix of mutual information by widow size values for each subject
        - nboot: number of requested bootstrap iterations
    Output:
        - dictionaries of bootstrapped distributions of Ipast and Ifuture values
    '''
    # Get subject IDs and initialize dictionaries
    subs = pd.unique(dat['Subject'])
    ip_boot_mi = {}
    if_boot_mi = {}
    
    # Loop through each subject and get bootstrapped estimates
    for subi,sub in enumerate(subs):
        sdat = dat[dat['Subject'] == sub]   # Get subject data
        w = wips[sub].argmax()              # Get subject's maximum window size
        x = np.array(sdat['Bead'])           # Get tones, responses, and sources for the subject
        r = np.array(sdat['Resp'])
        z = np.array(sdat['Jar'])
        xw,rw = ut.get_windowed_x(x,r,w=(w+1)) # Window the tones and responses appropriately
        ip_boot_mi[sub] = np.zeros(nboot)      # Initialize arrays of mutual information values
        if_boot_mi[sub] = np.zeros(nboot)
        
        # Run bootstrap procedure
        for boot in np.arange(nboot):
            idx = np.random.choice(np.arange(len(xw)),size=len(xw),replace=True)    # Select random indicies with replacement
            ip_boot_mi[sub][boot] = ut.mutual_inf_nsb(xw[idx],rw[idx],[2**(w+1),2]) # Use indexed data to compute Ipast
            if_boot_mi[sub][boot] = ut.mutual_inf_nsb(r[idx],z[idx],[2,2])          # Use indexed data to compute Ifuture
    # Return dictionaries of Ipast and Ifuture distributions
    return(ip_boot_mi,if_boot_mi)



def deltaBound(ib_ipast,ib_ifuture,p_ipast,p_ifuture):
    ''' 
    Function to calculate vertical distance from the bound between an empirical IB and participant predictive info
    ib_ipast: ipast of empirical IB (x of convex hull)
    ib_ifuture: ifuture of empirical IB (y of convex hull)
    p_ipast: participant ipast (uncorrected)
    p_ifuture: participant ifuture (uncorrected)
    
    Returns participant ifuture minus the empirical bound (more negative = farther away from the bound)
    '''
    ind = np.argwhere(np.array(ib_ipast) > p_ipast)[0][0]
    slp = (ib_ifuture[ind]-ib_ifuture[ind-1])/(ib_ipast[ind]-ib_ipast[ind-1])
    intercept = ib_ifuture[ind]-(slp*ib_ipast[ind])
    #Return distance between participant Ifuture and interpolated bound
    return p_ifuture - ((p_ipast*slp)+intercept)